import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


DEFAULT_MODEL_DIR = "outputs/qwen_te_translator_0_5b_lora"
DEFAULT_DATASET_PATH = "hf_data/telegraph_ft_nl2te"


def extract_assistant_reference(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def build_prompt_messages(messages):
    prompt = []
    for msg in messages:
        if msg.get("role") == "assistant":
            break
        prompt.append(msg)
    return prompt


def normalize_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def simple_line_overlap(prediction, reference):
    pred_lines = set(normalize_lines(prediction))
    ref_lines = set(normalize_lines(reference))
    if not ref_lines:
        return 0.0
    return len(pred_lines & ref_lines) / len(ref_lines)


def simple_te_format_score(prediction):
    lines = normalize_lines(prediction)
    if not lines:
        return 0.0
    tagged_lines = sum(
        1 for line in lines
        if ":" in line and line.split(":", 1)[0].strip().replace("-", "").replace(" ", "").isupper()
    )
    return tagged_lines / len(lines)


def generate_prediction(model, tokenizer, messages, max_new_tokens=512):
    prompt_messages = build_prompt_messages(messages)
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    new_ids = generated_ids[:, model_inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--dataset-path", default=os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    split = dataset[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    rng = random.Random(args.seed)
    indices = list(range(len(split)))
    rng.shuffle(indices)
    indices = indices[: args.num_samples]

    results = []
    total_overlap = 0.0
    total_format = 0.0

    for idx in indices:
        example = split[idx]
        reference = extract_assistant_reference(example["messages"])
        prediction = generate_prediction(
            model,
            tokenizer,
            example["messages"],
            max_new_tokens=args.max_new_tokens,
        )
        overlap = simple_line_overlap(prediction, reference)
        format_score = simple_te_format_score(prediction)
        total_overlap += overlap
        total_format += format_score

        result = {
            "index": idx,
            "reference": reference,
            "prediction": prediction,
            "line_overlap": overlap,
            "format_score": format_score,
        }
        results.append(result)

    summary = {
        "model_dir": args.model_dir,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_samples": len(results),
        "avg_line_overlap": total_overlap / len(results) if results else 0.0,
        "avg_format_score": total_format / len(results) if results else 0.0,
        "samples": results,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
