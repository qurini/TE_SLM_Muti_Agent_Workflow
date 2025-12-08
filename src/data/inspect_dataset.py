import json
from pathlib import Path
from collections import Counter

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON error in {path}: {e}")
    return data

def main():
    base = Path("trainingdata/compressed_output/telegraph-ft")
    train_path = base / "full_train.jsonl"
    val_path   = base / "full_val.jsonl"

    train = read_jsonl(train_path)
    val   = read_jsonl(val_path)

    print(f"Train examples: {len(train)}")
    print(f"Val examples:   {len(val)}")

    # Sanity check: roles
    role_sets = Counter()
    for ex in train[:100]:  # sample
        roles = tuple(m["role"] for m in ex["messages"])
        role_sets[roles] += 1

    print("Common role patterns in first 100 examples:")
    for roles, count in role_sets.most_common():
        print(f"  {roles}: {count}")

    # Show a sample
    print("\nSample example:")
    ex = train[0]
    for m in ex["messages"]:
        print(f"[{m['role'].upper()}]\n{m['content'][:300]}...\n")

if __name__ == "__main__":
    main()