from datasets import Dataset, DatasetDict, load_dataset
import json
from pathlib import Path

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def build_dataset():
    base = Path("trainingdata/compressed_output/telegraph-ft")
    train_data = list(read_jsonl(base / "full_train.jsonl"))
    val_data   = list(read_jsonl(base / "full_val.jsonl"))

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