"""Build the reference-embedding cache that lets LatentRAG train on an 8 GB GPU.

Training normally keeps a frozen copy of Qwen3-Embedding-0.6B on the GPU just to
produce target vectors for subqueries and passages. This script computes those
targets once, so training can drop that model (~1.2 GB plus its activations):

  1. passages:   copied from the released index shards (row = chunk_id; same
                 frozen encoder, fp32 compute, fp16 storage), CPU only
  2. subqueries: encoded with the exact encoder train.py would build (bf16, GPU)
  3. verify:     re-encodes a sample of passages with that encoder and checks
                 they match the shard rows, and that the teacher's ranking
                 distribution (temperature 0.03) is unchanged

Run from ~/LatentRAG with env.fish sourced, while nothing else uses the GPU:
    python build_reference_cache.py
Output: data/reference_cache/subset20k_qwen3-embedding-0.6b/ (~1 GB)
"""
import argparse, glob, json, os, random, sys, time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", default="data/reasoning_steps_subset20k",
               help="folder of training trajectory files (*/train/*.json)")
p.add_argument("--shards", default="data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard")
p.add_argument("--out", default="data/reference_cache/subset20k_qwen3-embedding-0.6b")
p.add_argument("--config-name", default="latentrag_qwen3-0.6b-base")
p.add_argument("--retriever", default="qwen3-embedding-0.6b")
p.add_argument("--gpu-env", default="latentrag_qwen3-0.6b-base_qwen3-embedding-0.6b/gpu_1_8g")
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--verify", type=int, default=2000, help="passages to re-encode for the check (0 = skip)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

if not os.path.isdir("src") or not os.path.isdir("config"):
    sys.exit("Run this from ~/LatentRAG")
sys.path.insert(0, os.path.abspath("src"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- 1. collect
files = sorted(glob.glob(os.path.join(args.src, "*", "train", "*.json")))
if not files:
    sys.exit(f"No training files under {args.src}/*/train/")
subqueries, chunk_ids, sample_pool = set(), set(), []
for f in files:
    for x in json.load(open(f))["results"]:
        # same processing as dataset.py: drop None, strip
        subqueries.update(s.strip() for s in x["subqueries"] if s is not None)
        for group in x["chunk_items"] + x["hard_neg_chunk_items"]:
            for c in group:
                chunk_ids.add(int(c["chunk_id"]))
                if c.get("text"):
                    sample_pool.append((int(c["chunk_id"]), c["text"]))
    log(f"read {f}")
subqueries = sorted(subqueries)
ids = np.array(sorted(chunk_ids), dtype=np.int64)
log(f"{len(subqueries):,} unique subqueries, {len(ids):,} unique passages")
os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------- 2. passages
paths = sorted(glob.glob(os.path.join(args.shards, "shard_index_*.npy")))
if not paths:
    sys.exit(f"No shard_index_*.npy files in {args.shards}")
shards = [np.load(f, mmap_mode="r") for f in paths]
offsets = np.cumsum([0] + [s.shape[0] for s in shards])
dim = shards[0].shape[1]
if ids[-1] >= offsets[-1]:
    sys.exit(f"chunk id {ids[-1]} is beyond the index ({offsets[-1]:,} rows)")

emb_path = os.path.join(args.out, "passage_emb.npy")
passage_emb = np.lib.format.open_memmap(emb_path + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(ids), dim))
shard_of = np.searchsorted(offsets, ids, side="right") - 1
for i in np.unique(shard_of):
    sel = np.nonzero(shard_of == i)[0]          # ids are sorted, so rows come out in order
    passage_emb[sel] = shards[i][ids[sel] - offsets[i]]
    if i % 20 == 0:
        log(f"  passages: shard {i + 1}/{len(shards)}")
passage_emb.flush()
del passage_emb
os.replace(emb_path + ".tmp.npy", emb_path)
np.save(os.path.join(args.out, "passage_ids.npy"), ids)
passage_emb = np.load(emb_path, mmap_mode="r")
log(f"wrote {len(ids):,} passage vectors ({os.path.getsize(emb_path) / 1e9:.2f} GB)")

# ---------------------------------------------------------------- 3. subqueries
import torch
from hydra import compose, initialize_config_dir
from retriever.encoder import Encoder

if not torch.cuda.is_available():
    sys.exit("The encoder needs the GPU (encoder.py sends inputs to cuda)")
with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name=args.config_name,
                  overrides=[f"retriever={args.retriever}", f"gpu_env={args.gpu_env}"])
# exactly what train.py builds for the frozen reference encoder
encoder = Encoder.build(cfg, trainable=False, use_lora=False)
encoder.base_model.to("cuda")
encoder.eval()
log(f"reference encoder: {cfg.retriever.encoder_name_or_path}, {next(encoder.parameters()).dtype}, "
    f"attn {cfg.retriever.attn_implementation}, max_length {cfg.retriever.encoder_query_max_length}")


def encode(texts, is_query):
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # less padding
    for b in range(0, len(texts), args.batch_size):
        batch = [texts[i] for i in order[b:b + args.batch_size]]
        with torch.no_grad():
            e = encoder.single_batch_encode(batch, is_query=is_query, return_tensors=True)
        out.append(e.float().cpu())
    emb = torch.cat(out) if out else torch.empty(0, dim)
    result = torch.empty_like(emb)
    result[torch.tensor(order, dtype=torch.long)] = emb
    return result


# dummy inputs the model substitutes when a trajectory has no subquery (latentrag.py)
dummy_subquery = "dummy subquery"
if dummy_subquery not in subqueries:
    subqueries.append(dummy_subquery)
t0 = time.time()
sub_emb = encode(subqueries, is_query=True)
np.save(os.path.join(args.out, "subquery_emb.npy"), sub_emb.numpy().astype(np.float32))
json.dump(subqueries, open(os.path.join(args.out, "subqueries.json"), "w"))
log(f"encoded {len(subqueries):,} subqueries in {time.time() - t0:.0f} s")

dummy_ids = [f"dummy_pos_chunk_id_{i}" for i in range(32)] + [f"dummy_neg_chunk_id_{i}" for i in range(32)]
dummy_texts = [f"dummy pos chunk text {i}" for i in range(32)] + [f"dummy neg chunk text {i}" for i in range(32)]
dummy_emb = encode(dummy_texts, is_query=False)
torch.save({k: dummy_emb[i].clone() for i, k in enumerate(dummy_ids)}, os.path.join(args.out, "extra_passages.pt"))

# ---------------------------------------------------------------- 4. verify
check = {}
if args.verify:
    rng = random.Random(args.seed)
    sample = rng.sample(sample_pool, min(args.verify, len(sample_pool)))
    fresh = encode([t for _, t in sample], is_query=False)
    rows = torch.from_numpy(np.asarray(passage_emb[np.searchsorted(ids, [c for c, _ in sample])], dtype=np.float32))
    cos = torch.nn.functional.cosine_similarity(fresh, rows, dim=-1)
    check = {
        "passages_checked": len(sample),
        "cosine_mean": round(cos.mean().item(), 5),
        "cosine_p01": round(cos.quantile(0.01).item(), 5),
        "cosine_min": round(cos.min().item(), 5),
        "below_0.98": int((cos < 0.98).sum()),
    }
    # Does using shard rows change the teacher's target distribution? Score each subquery
    # against 18 passages (fresh vs cached vectors) at the training temperature.
    temp = float(cfg.custom_training.temperature)
    q = torch.nn.functional.normalize(sub_emb[torch.randint(len(sub_emb) - 1, (len(sample) // 18,))], dim=-1)
    n = (len(sample) // 18) * 18
    pf = torch.nn.functional.normalize(fresh[:n], dim=-1).view(-1, 18, dim)
    pc = torch.nn.functional.normalize(rows[:n], dim=-1).view(-1, 18, dim)
    lf = torch.einsum("bd,bkd->bk", q, pf) / temp
    lc = torch.einsum("bd,bkd->bk", q, pc) / temp
    kl = torch.sum(torch.softmax(lf, -1) * (torch.log_softmax(lf, -1) - torch.log_softmax(lc, -1)), -1)
    check.update({"teacher_kl_mean": round(kl.mean().item(), 5),
                  "teacher_top1_agree": round((lf.argmax(-1) == lc.argmax(-1)).float().mean().item(), 4)})
    log(f"check: {check}")
    if check["cosine_mean"] < 0.99:
        log("WARNING: shard rows don't match the reference encoder closely; don't train on this cache")

json.dump({
    "source_files": files,
    "shards": args.shards,
    "num_passages": int(len(ids)),
    "num_subqueries": len(subqueries),
    "dim": int(dim),
    "encoder": cfg.retriever.encoder_name_or_path,
    "encoder_dtype": str(next(encoder.parameters()).dtype),
    "encoder_max_length": int(cfg.retriever.encoder_query_max_length),
    "check": check,
    "built": time.strftime("%Y-%m-%d %H:%M:%S"),
}, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
log(f"done: {args.out}")
