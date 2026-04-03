import json
import os
from pathlib import Path
import importlib.metadata

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DATA_PATH = Path("trainingdata/compressed_output/telegraph-ft/full_train.jsonl")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

def bitsandbytes_available():
    try:
        importlib.metadata.version("bitsandbytes")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find training data at {DATA_PATH}")

    ds = read_jsonl_basic(DATA_PATH)
    print(f"Loaded {len(ds)} training samples.")
    sample = ds[0]
    print("Keys:", sample.keys())
    print("First sample messages:\n", sample["messages"][:3])

    # 2) Load tokenizer & model
    print("\nLoading model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
    }
    if bitsandbytes_available():
        print("bitsandbytes detected: trying 4-bit model load.")
        load_kwargs.update({
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        })
    else:
        print("bitsandbytes not found: falling back to non-quantized model load.")
        load_kwargs["torch_dtype"] = torch.bfloat16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            **load_kwargs,
        )
    except Exception as e:
        print(f"Model load failed with current settings: {e}")
        print("Falling back to tokenizer-only sanity check.")
        model = None

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
    if model is not None:
        print("Model load sanity check: OK")
    else:
        print("Tokenizer/chat-template sanity check: OK (model load skipped)")

if __name__ == "__main__":
    main()
