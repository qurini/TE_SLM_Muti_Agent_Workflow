# TE_SLM_Muti_Agent_Workflow

Initial focus: Paper B pilot for a tiny `NL -> TE` translator used inside a TE-based multi-agent workflow.

## First milestone

Train a small Qwen-based LoRA model on the existing chat-formatted `NL -> TE` dataset, then use that model in the first pipeline experiment without a `TE -> NL` translator.

## Quick start

1. Inspect the dataset:

```bash
python src/data/inspect_dataset.py
```

2. Build the Hugging Face dataset:

```bash
python src/data/build_hf_dataset.py
```

3. Run the model sanity check:

```bash
python src/training/sanity_check_qwen_te.py
```

4. Train the first LoRA pilot:

```bash
bash scripts/runpod_first_pilot.sh
```

## RunPod

See [docs/RUNPOD_SETUP.md](/F:/TE/TE_SLM_Muti_Agent_Workflow/docs/RUNPOD_SETUP.md) for the first remote setup flow.
