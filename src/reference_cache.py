"""Precomputed reference-encoder targets for LatentRAG training (Qwen3-0.6B port).

The frozen reference encoder only turns subquery texts and passage texts into
fixed target vectors. On an 8 GB GPU, keeping that third 0.6B model resident
runs out of memory, so build_reference_cache.py computes the targets once and
training reads them from here instead.

- Passages: rows of the released wiki-18 index shards (the same frozen
  Qwen3-Embedding-0.6B, computed in fp32 and stored in fp16), keyed by chunk_id.
- Subqueries: encoded once by the same frozen encoder the trainer would build.

Files in the cache folder:
    passage_ids.npy      int64, sorted chunk ids
    passage_emb.npy      float16 [n_passages, dim], row i belongs to passage_ids[i]
    subqueries.json      list of subquery strings (after .strip(), as the loader does)
    subquery_emb.npy     float32 [n_subqueries, dim]
    extra_passages.pt    {chunk_id: tensor} for the model's no-subquery dummy chunks
    meta.json            what it was built from
"""
import json
import os

import numpy as np
import torch


class ReferenceEmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        need = ["passage_ids.npy", "passage_emb.npy", "subqueries.json", "subquery_emb.npy", "meta.json"]
        missing = [f for f in need if not os.path.isfile(os.path.join(path, f))]
        if missing:
            raise FileNotFoundError(
                f"Reference cache at {path} is missing {missing}; build it with build_reference_cache.py "
                f"or set retriever.reference_cache_path=null to use the reference encoder on the GPU")
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.passage_ids = np.load(os.path.join(path, "passage_ids.npy"))
        self.passage_emb = np.load(os.path.join(path, "passage_emb.npy"), mmap_mode="r")
        texts = json.load(open(os.path.join(path, "subqueries.json")))
        self.subquery_index = {t: i for i, t in enumerate(texts)}
        self.subquery_emb = np.load(os.path.join(path, "subquery_emb.npy"), mmap_mode="r")
        extra_path = os.path.join(path, "extra_passages.pt")
        self.extra_passages = torch.load(extra_path, weights_only=True) if os.path.isfile(extra_path) else {}

    def __repr__(self):
        return (f"ReferenceEmbeddingCache({self.path}: {len(self.passage_ids):,} passages, "
                f"{len(self.subquery_index):,} subqueries)")

    def subqueries(self, texts, device) -> torch.Tensor:
        try:
            idx = [self.subquery_index[t] for t in texts]
        except KeyError as e:
            raise KeyError(f"Subquery {e.args[0]!r} is not in the reference cache {self.path}; "
                           f"rebuild it from the training files you are using") from None
        rows = np.asarray(self.subquery_emb[np.asarray(idx)], dtype=np.float32)
        return torch.from_numpy(rows).to(device)

    def passages(self, chunk_ids, device) -> torch.Tensor:
        out = [None] * len(chunk_ids)
        num_pos, num_rows = [], []
        for i, cid in enumerate(chunk_ids):
            key = str(cid)
            if key in self.extra_passages:
                out[i] = self.extra_passages[key].float()
            else:
                num_pos.append(i)
                num_rows.append(int(key))
        if num_rows:
            ids = np.asarray(num_rows, dtype=np.int64)
            j = np.searchsorted(self.passage_ids, ids)
            bad = (j >= len(self.passage_ids)) | (self.passage_ids[np.minimum(j, len(self.passage_ids) - 1)] != ids)
            if bad.any():
                raise KeyError(f"Passage {int(ids[bad][0])} is not in the reference cache {self.path}; "
                               f"rebuild it from the training files you are using")
            rows = torch.from_numpy(np.asarray(self.passage_emb[j], dtype=np.float32))
            for k, i in enumerate(num_pos):
                out[i] = rows[k]
        return torch.stack(out).to(device)
