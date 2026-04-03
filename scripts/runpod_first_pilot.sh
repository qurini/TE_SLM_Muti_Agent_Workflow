#!/usr/bin/env bash
set -euo pipefail

python src/data/inspect_dataset.py
python src/training/sanity_check_qwen_te.py
python src/data/build_hf_dataset.py
python src/training/train_qwen_te_translator.py
python src/eval/eval_translator.py --num-samples 10 --output-json outputs/qwen_te_translator_0_5b_lora/eval_summary.json
