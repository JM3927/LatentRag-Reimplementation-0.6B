"""Cut the training subset (and a correct-only full copy) from the released
Search-R1 trajectories, one file at a time so RAM stays low.

Run from ~/LatentRAG:
    python make_subsets.py            # 20k subset + correct-only full copy
    python make_subsets.py --no-full  # 20k subset only
"""
import argparse, gc, glob, json, os, random

p = argparse.ArgumentParser()
p.add_argument("--src", default="data/reasoning_steps")
p.add_argument("--per-file", type=int, default=10250)  # x2 files ~= 20k train + 256 held out
p.add_argument("--seed", type=int, default=42)
p.add_argument("--corpus", default="data/corpus/wiki18_100w.jsonl")
p.add_argument("--no-full", action="store_true")
args = p.parse_args()

subset_dir = args.src + "_subset20k"
full_dir = args.src + "_correct"
files = sorted(glob.glob(os.path.join(args.src, "*", "train", "*.json")))
if not files:
    raise SystemExit(f"No train files under {args.src}/*/train/")

REQUIRED = ["query", "thoughts", "subqueries", "answer", "chunk_items",
            "hard_neg_chunk_items", "em_score", "force_answer"]

def well_formed(x):
    if any(k not in x for k in REQUIRED):
        return False
    subs = [s for s in x["subqueries"] if s is not None]
    return len(x["thoughts"]) - 1 == len(subs) == len(x["chunk_items"])

def missing_text(items):
    return sum(1 for x in items for group in (x["chunk_items"] + x["hard_neg_chunk_items"])
               for c in group if not c.get("text"))

def fill_text(items):
    """Fill missing passage text by streaming the corpus once (never loads it whole)."""
    need = {str(c["chunk_id"]) for x in items for group in (x["chunk_items"] + x["hard_neg_chunk_items"])
            for c in group if not c.get("text")}
    print(f"  filling text for {len(need):,} passages from {args.corpus}")
    found = {}
    with open(args.corpus) as fh:
        for line in fh:
            obj = json.loads(line)
            if str(obj["id"]) in need:
                found[str(obj["id"])] = obj["contents"]
    for x in items:
        for group in x["chunk_items"] + x["hard_neg_chunk_items"]:
            for c in group:
                if not c.get("text"):
                    c["text"] = found[str(c["chunk_id"])]

def write(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"results": results}, fh)

for f in files:
    rel = os.path.relpath(f, args.src)
    with open(f) as fh:
        data = json.load(fh)["results"]
    correct = [x for x in data if x.get("em_score") == 1.0 and well_formed(x)]
    del data; gc.collect()
    print(f"{rel}: {len(correct):,} correct, well-formed trajectories")

    rng = random.Random(args.seed)
    subset = rng.sample(correct, min(args.per_file, len(correct)))
    if missing_text(subset):
        fill_text(subset)
    write(subset, os.path.join(subset_dir, rel))
    print(f"  wrote {len(subset):,} to {os.path.join(subset_dir, rel)}")

    if not args.no_full:
        if missing_text(correct):
            fill_text(correct)
        write(correct, os.path.join(full_dir, rel))
        print(f"  wrote {len(correct):,} to {os.path.join(full_dir, rel)}")
    del correct, subset; gc.collect()

print("done")
