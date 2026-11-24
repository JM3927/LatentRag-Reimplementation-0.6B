import glob, json, sys

split = sys.argv[1] if len(sys.argv) > 1 else "test"
if split == "train":
    files = sorted(glob.glob("data/reasoning_steps/*/train/*.json"))
else:
    files = sorted(glob.glob("data/reasoning_steps/*/test/*.json") + glob.glob("data/reasoning_steps/*/dev/*.json"))

if not files:
    print(f"No {split} trajectory files found under data/reasoning_steps")

for f in files:
    with open(f) as fh:
        results = json.load(fh)["results"]
    correct = sum(1 for x in results if x.get("em_score") == 1.0)
    print(f"{f}: {len(results)} questions, teacher EM {correct / len(results):.4f}")

if files:
    print("fieldsL:", sorted(results[0].keys()))
