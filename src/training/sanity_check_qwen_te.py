import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_PATH = Path("data/telegraph_ft/full_train.jsonl")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def read_jsonl_basic(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("JSON error:", e, "line:", line[:120])
    return data

def main():
    # 1) Load a few samples
    ds = read_jsonl_basic(DATA_PATH)
    print(f"Loaded {len(ds)} training samples.")
    sample = ds[0]
    print("Keys:", sample.keys())
    print("First sample messages:\n", sample["messages"][:3])

    # 2) Load tokenizer & model
    print("\nLoading model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        load_in_4bit=True,  # requires bitsandbytes; change to False if needed
    )

    # 3) Try applying chat template to one example
    chat = sample["messages"]
    text = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
    )
    print("\nChat template output preview:\n", text[:400])

    # 4) Tokenize to ensure it fits
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    print("\nTokenized length:", inputs["input_ids"].shape)

if __name__ == "__main__":
    main()
