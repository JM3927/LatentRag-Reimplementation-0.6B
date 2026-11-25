"""Recall check for the IVF-PQ retriever against exact search (Qwen3-0.6B port).

Ground truth: brute-force inner product over all 21M normalized shard vectors on the GPU
(what the original TorchRetriever returns). Compares IVFPQRetriever top-k at several
nprobe values, with and without exact re-ranking. Pass mark from the action list:
recall@3 >= 0.95 at the server's settings.

  python check_retriever_recall.py --n 500
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from hydra import compose, initialize_config_dir  # noqa: E402

from retriever.encoder import Encoder  # noqa: E402
from retriever.ivfpq_retriever import IVFPQRetriever  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=500, help="number of queries")
p.add_argument("--k", type=int, default=3)
p.add_argument("--nprobes", default="256,512,1024,2048")
p.add_argument("--candidates", default=None,
               help="re-rank candidate counts to sweep (default: the config value)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="data/corpus/ivfpq/recall_check.json")
args = p.parse_args()

with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name="retriever_server_ivfpq")

# ---- queries: questions from the eval sets, equal share each ---------------------------
sources = ["nq/test", "hotpotqa/dev", "2wikimultihopqa/dev", "musique/dev", "popqa/test"]
rng = random.Random(args.seed)
queries = []
for s in sources:
    with open(f"data/qa_data/{s}.jsonl") as f:
        qs = [json.loads(line)["question"] for line in f]
    queries += rng.sample(qs, args.n // len(sources))
print(f"{len(queries)} queries from {', '.join(sources)}")

ret = IVFPQRetriever(cfg)
ret.set_encoder(Encoder.load(cfg, trainable=False, use_lora=False))
q = ret._unit(ret.encode_queries(queries))
dev = "cuda"
ret.encoder.base_model.to("cpu")  # free GPU for the brute-force pass
torch.cuda.empty_cache()

# ---- exact top-k over all shards --------------------------------------------------------
K = 10
t = time.time()
qt = torch.from_numpy(q).to(dev)
best_s = torch.full((len(q), K), -float("inf"), device=dev)
best_i = torch.full((len(q), K), -1, dtype=torch.long, device=dev)
for si, shard in enumerate(ret.vectors.shards):
    x = torch.from_numpy(np.ascontiguousarray(shard)).to(dev).float()
    x /= x.norm(dim=1, keepdim=True)
    s, i = (qt @ x.T).topk(K, dim=1)
    s = torch.cat([best_s, s], 1)
    i = torch.cat([best_i, i + int(ret.vectors.starts[si])], 1)
    best_s, o = s.topk(K, dim=1)
    best_i = i.gather(1, o)
    if si % 20 == 0:
        print(f"  exact search: shard {si + 1}/{len(ret.vectors.shards)} ({time.time() - t:.0f}s)", flush=True)
ex_s, ex_i = best_s.cpu().numpy(), best_i.cpu().numpy()
print(f"exact search done in {time.time() - t:.0f}s")

# ---- compare ----------------------------------------------------------------------------
k = args.k
kth = ex_s[:, k - 1:k]


def recall(ids, scores):
    """id recall@k, and tie-tolerant recall (counts a hit if its exact score >= true k-th
    score - 1e-4; the corpus has duplicate passages with identical vectors)."""
    id_r = np.mean([len(set(a) & set(b)) / k for a, b in zip(ids[:, :k], ex_i[:, :k])])
    tie_r = np.mean(np.minimum((scores[:, :k] >= kth - 1e-4).sum(1), k) / k)
    return float(id_r), float(tie_r)


rows = []
cands = [int(x) for x in args.candidates.split(",")] if args.candidates else [ret.rerank_candidates]
default_cand = ret.rerank_candidates
settings = [(True, c) for c in cands] + [(False, 0)]
for rerank, cand in settings:
    ret.rerank_candidates = cand or default_cand
    for nprobe in [int(x) for x in args.nprobes.split(",")]:
        ret.search(q[:8], k, nprobe=nprobe, rerank=rerank)  # warm-up
        t = time.time()
        s, i = ret.search(q, k, nprobe=nprobe, rerank=rerank)
        ms = (time.time() - t) / len(q) * 1000
        if not rerank:  # PQ scores are approximate; recompute exact ones for the tie check
            s = np.einsum("nkd,nd->nk", ret.vectors.get(i.ravel()).reshape(len(q), k, -1), q)
        id_r, tie_r = recall(i, s)
        top1 = float(np.mean(i[:, 0] == ex_i[:, 0]))
        rows.append(dict(rerank=rerank, candidates=cand, nprobe=nprobe, recall_at_k=id_r,
                         recall_at_k_ties=tie_r, top1_match=top1, ms_per_query=ms))
        print(f"rerank={str(rerank):5} cand={cand:5} nprobe={nprobe:5}  recall@{k} {id_r:.4f} "
              f"(ties ok {tie_r:.4f})  top-1 match {top1:.4f}  {ms:.1f} ms/query", flush=True)

ret.rerank_candidates = default_cand
server = next((r for r in rows if r["rerank"] and r["nprobe"] == ret.nprobe
               and r["candidates"] == default_cand), None)
if server is None:
    print("\n(the server's own nprobe/candidates were not in this sweep; no verdict)")
    with open(args.out, "w") as f:
        json.dump(dict(n_queries=len(q), k=k, sources=sources, seed=args.seed, rows=rows), f, indent=2)
    sys.exit(0)
verdict = "PASS" if server["recall_at_k"] >= 0.95 else "FAIL"
print(f"\nServer settings (nprobe {ret.nprobe}, re-rank {ret.rerank_candidates}): "
      f"recall@{k} {server['recall_at_k']:.4f} -> {verdict} (pass mark 0.95)")
with open(args.out, "w") as f:
    json.dump(dict(n_queries=len(q), k=k, sources=sources, seed=args.seed, rows=rows,
                   verdict=verdict), f, indent=2)
print(f"wrote {args.out}")
