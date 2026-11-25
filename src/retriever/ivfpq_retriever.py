"""IVF-PQ retriever with exact re-ranking, for machines that cannot hold the full index.

Qwen3-0.6B port (not in the original repo). The original TorchRetriever keeps all 21M
fp16 vectors (43 GB) on GPU/CPU and the whole corpus (14 GB jsonl) as Python dicts.
This one keeps:
  - the FAISS IVF-PQ index built by build_ivfpq.py (2.9 GB RAM) for candidate search,
  - the original fp16 shards memory-mapped from disk, for exact re-scoring of the
    candidates (same scores as TorchRetriever: inner product of L2-normalized vectors),
  - a byte-offset table into the corpus jsonl (168 MB), so passage text is read on demand.

Outputs (ChunkItem / RetrieverOutput) and chunk ids (= corpus line = index row) match
TorchRetriever, so it can sit behind the same REST server.
"""
import glob
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import torch

from retriever.torch_retriever_gpu import ChunkItem, RetrieverOutput

logger = logging.getLogger(__name__)


def build_line_offsets(corpus_file, out_path, block=256 << 20):
    """Byte offset of every line start in corpus_file, plus the file end (n + 1 entries)."""
    logger.info(f"Building line offsets for {corpus_file} (one-time, reads the whole file)")
    size = os.path.getsize(corpus_file)
    starts = [np.zeros(1, dtype=np.int64)]
    pos = 0
    with open(corpus_file, "rb") as f:
        while True:
            buf = f.read(block)
            if not buf:
                break
            nl = np.flatnonzero(np.frombuffer(buf, dtype=np.uint8) == 10).astype(np.int64)
            starts.append(nl + pos + 1)
            pos += len(buf)
    offsets = np.concatenate(starts)
    if offsets[-1] != size:  # last line without trailing newline
        offsets = np.append(offsets, size)
    tmp = out_path + ".tmp.npy"
    np.save(tmp, offsets)
    os.replace(tmp, out_path)
    logger.info(f"Wrote {out_path}: {len(offsets) - 1:,} lines")
    return offsets


class CorpusReader:
    """Random access to the 'contents' field of the corpus jsonl by line number."""

    def __init__(self, corpus_file, offsets_path):
        if os.path.exists(offsets_path):
            self.offsets = np.load(offsets_path)
        else:
            self.offsets = build_line_offsets(corpus_file, offsets_path)
        if self.offsets[-1] != os.path.getsize(corpus_file):
            raise ValueError(f"{offsets_path} does not match {corpus_file}; delete it to rebuild")
        self.fd = os.open(corpus_file, os.O_RDONLY)

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, i):
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        return json.loads(os.pread(self.fd, end - start, start))["contents"]


class ShardVectors:
    """The released fp16 embedding shards on disk; rows L2-normalized on read.

    Random rows are fetched with os.pread from a thread pool, with kernel readahead off.
    (Fancy-indexing the memory maps instead faults pages in one at a time, each pulling
    ~128 KB of readahead for a 2 KB row: ~13 s for 19k rows vs well under 1 s this way.)
    `shards` (memory maps) remain available for sequential scans.
    """

    def __init__(self, folder, io_threads=32):
        paths = [p for p in glob.glob(os.path.join(folder, "shard_index_*.npy"))
                 if re.search(r"shard_index_\d{4}\.npy$", p)]
        if not paths:
            raise FileNotFoundError(f"no shard_index_XXXX.npy files in {folder}")
        paths.sort(key=lambda p: int(re.search(r"(\d{4})\.npy$", p).group(1)))
        self.shards = [np.load(p, mmap_mode="r") for p in paths]
        for s in self.shards:
            if s.dtype != np.float16 or s.ndim != 2 or not s.flags.c_contiguous:
                raise ValueError(f"unexpected shard layout {s.dtype} {s.shape}")
        sizes = np.array([s.shape[0] for s in self.shards], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(sizes)])
        self.ntotal = int(self.starts[-1])
        self.dim = self.shards[0].shape[1]
        self.row_bytes = self.dim * 2
        self.header = [int(s.offset) for s in self.shards]
        self.fds = [os.open(p, os.O_RDONLY) for p in paths]
        for fd in self.fds:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.io_threads = io_threads
        self.pool = ThreadPoolExecutor(io_threads)

    def _read_rows(self, shard_of, rows):
        rb = self.row_bytes
        return b"".join(os.pread(self.fds[s], rb, self.header[s] + r * rb)
                        for s, r in zip(shard_of.tolist(), rows.tolist()))

    def get(self, ids):
        """ids: 1-D int64 array -> float32 (len(ids), dim), unit-norm rows, same order."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        shard_of = np.searchsorted(self.starts, sorted_ids, side="right") - 1
        rows = sorted_ids - self.starts[shard_of]
        parts = np.array_split(np.arange(len(ids)), min(self.io_threads, len(ids)))
        bufs = self.pool.map(lambda p: self._read_rows(shard_of[p], rows[p]), parts)
        sorted_vecs = np.frombuffer(b"".join(bufs), dtype=np.float16).reshape(len(ids), self.dim)
        out = np.empty((len(ids), self.dim), dtype=np.float32)
        out[order] = sorted_vecs
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out


class IVFPQRetriever:

    def __init__(self, config):
        r, ivf = config.retriever, config.ivfpq
        self.retriever_tag = r.retriever_tag
        self.top_k = r.top_k
        self.batch_size = r.retrieval_batch_size
        self.nprobe = int(ivf.nprobe)
        self.rerank_candidates = int(ivf.rerank_candidates)
        self.lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool
        self.encoder = None

        if ivf.get("omp_threads"):
            faiss.omp_set_num_threads(int(ivf.omp_threads))

        marker = os.path.join(os.path.dirname(ivf.index_path), "NORMALIZED")
        if not os.path.exists(marker):
            raise RuntimeError(f"{marker} missing: the index was built from unnormalized vectors "
                               f"by an old build_ivfpq.py. Rebuild it.")
        t = time.time()
        self.index = faiss.read_index(ivf.index_path)
        self.ivf = faiss.extract_index_ivf(self.index)
        self.ivf.nprobe = self.nprobe
        # Split each query's lists across threads (default 0 = one thread per query):
        # single queries 113 -> 18 ms at nprobe 2048, batches unchanged.
        self.ivf.parallel_mode = 1
        logger.info(f"Loaded {ivf.index_path}: {self.index.ntotal:,} vectors, "
                    f"nlist {self.ivf.nlist}, nprobe {self.nprobe} ({time.time() - t:.1f}s)")

        self.vectors = ShardVectors(os.path.join(r.corpus_path, r.corpus_index_folder))
        self.corpus = CorpusReader(os.path.join(r.corpus_path, r.corpus_file), ivf.offsets_path)
        if not (self.index.ntotal == self.vectors.ntotal == len(self.corpus)):
            raise ValueError(f"size mismatch: index {self.index.ntotal}, shards "
                             f"{self.vectors.ntotal}, corpus {len(self.corpus)}")
        logger.info(f"Corpus and shards: {len(self.corpus):,} passages; re-ranking top "
                    f"{self.rerank_candidates} candidates exactly")

    def get_retriever_tag(self):
        return self.retriever_tag

    def set_encoder(self, encoder):
        self.encoder = encoder
        self.encoder.eval()

    def get_encoder(self):
        return self.encoder

    # ---- core search -------------------------------------------------------------
    @staticmethod
    def _unit(q):
        q = np.ascontiguousarray(q, dtype=np.float32)
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    def search(self, q, top_k, nprobe=None, rerank=True):
        """q: (n, dim) float32, unit-norm. Returns (scores, ids), each (n, top_k), best first."""
        if nprobe is not None:
            self.ivf.nprobe = int(nprobe)
        try:
            if not rerank:
                return self.index.search(q, top_k)
            k_cand = max(self.rerank_candidates, top_k)
            _, cand = self.index.search(q, k_cand)
            n = q.shape[0]
            valid = cand >= 0
            flat = np.where(valid, cand, 0).ravel()
            uniq, inv = np.unique(flat, return_inverse=True)
            vecs = self.vectors.get(uniq)
            exact = np.einsum("nkd,nd->nk", vecs[inv].reshape(n, k_cand, -1), q)
            exact[~valid] = -np.inf
            top = np.argsort(-exact, axis=1, kind="stable")[:, :top_k]
            scores = np.take_along_axis(exact, top, axis=1)
            ids = np.take_along_axis(cand, top, axis=1)
            ids[~np.isfinite(scores)] = -1
            return scores, ids
        finally:
            self.ivf.nprobe = self.nprobe

    def _to_items(self, scores, ids):
        return [[ChunkItem(chunk_id=str(int(i)), text=self.corpus.get(int(i)), score=float(s))
                 for s, i in zip(srow, irow) if i != -1]
                for srow, irow in zip(scores, ids)]

    def _retrieve(self, q, top_k, start_time):
        results = []
        for b in range(0, len(q), self.batch_size):
            scores, ids = self.search(self._unit(q[b:b + self.batch_size]), top_k)
            results.extend(self._to_items(scores, ids))
        return RetrieverOutput(results=results, start_time=start_time, end_time=time.perf_counter())

    # ---- same interface as TorchRetriever --------------------------------------------
    @torch.inference_mode()
    def encode_queries(self, queries):
        if self.encoder is None:
            raise ValueError("No encoder loaded (ivfpq.load_encoder=false); text mode is disabled.")
        embs = [self.encoder.single_batch_encode(queries[b:b + self.batch_size], is_query=True)
                for b in range(0, len(queries), self.batch_size)]
        torch.cuda.empty_cache()
        return np.concatenate(embs, axis=0).astype(np.float32)

    def retrieve_with_text(self, queries, top_k=None):
        start_time = time.perf_counter()
        if isinstance(queries, str):
            queries = [queries]
        top_k = self.top_k if top_k is None else top_k
        if top_k == 0:
            return RetrieverOutput([[] for _ in queries], start_time, time.perf_counter())
        with self.lock:
            return self._retrieve(self.encode_queries(queries), top_k, start_time)

    def retrieve_with_embeddings(self, query_embeddings, top_k=None):
        """query_embeddings: (n, dim); normalized here too, so unnormalized input is fine."""
        start_time = time.perf_counter()
        top_k = self.top_k if top_k is None else top_k
        if top_k == 0:
            return RetrieverOutput([[] for _ in query_embeddings], start_time, time.perf_counter())
        with self.lock:
            return self._retrieve(np.asarray(query_embeddings, dtype=np.float32), top_k, start_time)
