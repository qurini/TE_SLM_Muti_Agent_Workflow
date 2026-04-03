from datasets import Dataset, DatasetDict, load_dataset
import json
import os
from pathlib import Path

COMPACT_SYSTEM_PROMPT = os.getenv(
    "COMPACT_SYSTEM_PROMPT",
    "You are a Telegraph English translator. Convert the user's text into Telegraph English. "
    "Return only TE lines. Preserve meaning, numbers, citations, and structure. "
    "Use concise uppercase TE with atomic lines and symbolic relations where appropriate."
)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def compact_messages(example):
    messages = example["messages"]
    compacted = []
    replaced_system = False
    for message in messages:
        if message["role"] == "system" and not replaced_system:
            compacted.append({
                "role": "system",
                "content": COMPACT_SYSTEM_PROMPT,
            })
            replaced_system = True
        else:
            compacted.append(message)
    example["messages"] = compacted
    return example

def build_dataset():
    base = Path("trainingdata/compressed_output/telegraph-ft")
    train_data = list(read_jsonl(base / "full_train.jsonl"))
    val_data   = list(read_jsonl(base / "full_val.jsonl"))

    train_data = [compact_messages(example) for example in train_data]
    val_data = [compact_messages(example) for example in val_data]

    # If you don't have a test set yet, we can just split val later.
    ds_train = Dataset.from_list(train_data)
    ds_val   = Dataset.from_list(val_data)

    dsd = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
    })

    dsd.save_to_disk("hf_data/telegraph_ft_nl2te")
    print(dsd)

if __name__ == "__main__":
    build_dataset()
