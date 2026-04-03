# RunPod Setup

This project's first pilot is a small `NL -> TE` LoRA fine-tune on Qwen.

## 1. Create the pod

- Use `Secure Cloud` and create a persistent pod.
- Recommended GPU for the first pilot: `A10G 24GB`, `L4 24GB`, or better.
- Recommended image: an official `PyTorch` + CUDA template.
- Attach at least `50GB` of disk if you want to keep datasets and checkpoints.

## 2. Clone and install

```bash
cd /workspace
git clone <YOUR_REPO_URL> te_slim_multi_agent
cd te_slim_multi_agent

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Verify the data exists

Expected training files:

```text
trainingdata/compressed_output/telegraph-ft/full_train.jsonl
trainingdata/compressed_output/telegraph-ft/full_val.jsonl
```

If they are not in the repo yet, upload them into that exact folder before training.

## 4. First sanity run

```bash
python src/training/sanity_check_qwen_te.py
```

Expected result:

- the dataset loads
- Qwen tokenizer/model loads
- the chat template renders your message list correctly
- tokenization completes without errors

## 5. Build the HF dataset

```bash
python src/data/build_hf_dataset.py
```

Expected output folder:

```text
hf_data/telegraph_ft_nl2te
```

## 6. Start the first training pilot

```bash
bash scripts/runpod_first_pilot.sh
```

This first run is only meant to validate:

- the dataset format
- LoRA wiring
- memory fit on your selected GPU
- the model can start learning the TE format

## 7. What to check after the run

- Training loss decreases.
- Validation runs complete.
- A checkpoint appears under:

```text
outputs/qwen_te_translator_0_5b_lora
```

- A qualitative evaluation summary appears under:

```text
outputs/qwen_te_translator_0_5b_lora/eval_summary.json
```

## 8. Next step after the pilot

Once the pilot is stable, we will:

1. add held-out inference evaluation
2. inspect qualitative TE outputs
3. move into the first benchmarked pipeline experiment
