"""Build the CPU IVF-PQ search index from the released fp16 index shards.
Run from ~/LatentRAG:  python build_ivfpq.py
Resumable: if it stops, rerun the same command and it continues.
Peak RAM is about 10 GB; the finished index is about 3 GB."""
import argparse, glob, os, time
import numpy as np
import faiss

p = argparse.ArgumentParser()
p.add_argument("--shards", default="data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard")
p.add_argument("--out", default="data/corpus/ivfpq")
p.add_argument("--nlist", type=int, default=16384)    # IVF lists
p.add_argument("--m", type=int, default=128)          # PQ sub-vectors -> 128-byte codes
p.add_argument("--train-size", type=int, default=1_000_000)
p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
p.add_argument("--save-every", type=int, default=20)  # shards between progress saves
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

faiss.omp_set_num_threads(args.threads)
os.makedirs(args.out, exist_ok=True)
trained_path = os.path.join(args.out, "trained.index")
partial_path = os.path.join(args.out, "partial.index")
final_path = os.path.join(args.out, f"wiki18_ivf{args.nlist}_pq{args.m}.index")
# The released shards are NOT unit length (norm ~104); the repo's retriever L2-normalizes
# them on load (torch_retriever_gpu.py), i.e. ranks by cosine. So do we. The marker keeps a
# rerun from resuming files written by the first version of this script, which didn't.
marker = os.path.join(args.out, "NORMALIZED")
if not os.path.exists(marker):
    if any(os.path.exists(f) for f in (trained_path, partial_path, final_path)):
        raise SystemExit(f"{args.out} holds files built from unnormalized vectors; delete that folder and rerun")
    open(marker, "w").write("vectors L2-normalized before training and adding\n")
if os.path.exists(final_path):
    raise SystemExit(f"Already built: {final_path}")


def unit(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    faiss.normalize_L2(a)   # in place
    return a

paths = sorted(glob.glob(os.path.join(args.shards, "shard_index_*.npy")))
if not paths:
    raise SystemExit(f"No shard_index_*.npy files in {args.shards}")
shards = [np.load(f, mmap_mode="r") for f in paths]   # memory-mapped: nothing loaded yet
sizes = [s.shape[0] for s in shards]
d, total = shards[0].shape[1], sum(sizes)
log(f"{len(paths)} shards, {total:,} vectors, dim {d}, {args.threads} threads")

# 1. Train the coarse quantizer and PQ codebooks on a random sample (or reuse a saved one).
if os.path.exists(trained_path):
    index = faiss.read_index(trained_path)
    log(f"loaded trained index from {trained_path}")
else:
    rng = np.random.default_rng(args.seed)
    take = np.sort(rng.choice(total, size=min(args.train_size, total), replace=False))
    offsets = np.cumsum([0] + sizes)
    parts = []
    for i, s in enumerate(shards):
        sel = take[(take >= offsets[i]) & (take < offsets[i + 1])] - offsets[i]
        if len(sel):
            parts.append(np.asarray(s[sel], dtype=np.float32))
    x = np.concatenate(parts)
    del parts
    log(f"training sample {x.shape}; raw mean vector norm {np.linalg.norm(x[:1000], axis=1).mean():.1f} (released shards: ~104)")
    x = unit(x)
    log(f"normalized: mean norm {np.linalg.norm(x[:1000], axis=1).mean():.3f} (expect 1.000)")
    index = faiss.index_factory(d, f"IVF{args.nlist},PQ{args.m}", faiss.METRIC_INNER_PRODUCT)
    faiss.extract_index_ivf(index).verbose = True
    log("training: k-means then PQ codebooks (the longest single step)")
    index.train(x)
    del x
    faiss.extract_index_ivf(index).verbose = False
    faiss.write_index(index, trained_path)
    log(f"saved {trained_path}")

# 2. Add every shard in order, so index id i = corpus line i. Resume from a partial index if one exists.
start = 0
if os.path.exists(partial_path):
    index = faiss.read_index(partial_path)
    done = np.cumsum([0] + sizes)
    matches = np.nonzero(done == index.ntotal)[0]
    if len(matches) != 1:
        raise SystemExit(f"{partial_path} holds {index.ntotal:,} vectors, which isn't a whole number of shards; delete it and rerun")
    start = int(matches[0])
    log(f"resuming at shard {start} with {index.ntotal:,} vectors already added")

for i in range(start, len(shards)):
    index.add(unit(shards[i]))
    if (i + 1) % args.save_every == 0 and i + 1 < len(shards):
        faiss.write_index(index, partial_path + ".tmp")
        os.replace(partial_path + ".tmp", partial_path)
        log(f"added {i + 1}/{len(shards)} shards, {index.ntotal:,} vectors (progress saved)")
    elif (i + 1) % 5 == 0:
        log(f"added {i + 1}/{len(shards)} shards, {index.ntotal:,} vectors")

assert index.ntotal == total, (index.ntotal, total)
faiss.write_index(index, final_path + ".tmp")
os.replace(final_path + ".tmp", final_path)
if os.path.exists(partial_path):
    os.remove(partial_path)
log(f"done: {final_path}, {os.path.getsize(final_path) / 1e9:.2f} GB, {index.ntotal:,} vectors")
