from datasets import load_from_disk
import os
import importlib.metadata
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_PATH = os.getenv("DATASET_PATH", "hf_data/telegraph_ft_nl2te")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/qwen_te_translator_0_5b_lora")
USE_4BIT = os.getenv("USE_4BIT", "1") == "1"
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "4"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))

def bitsandbytes_available():
    try:
        importlib.metadata.version("bitsandbytes")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

def formatting_func(example):
    """
    Turn the 'messages' list into a single chat string using the
    model's chat template. We do NOT add a generation prompt because
    we want the assistant content included for training.
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text

if __name__ == "__main__":
    # 1. Load dataset
    dataset = load_from_disk(DATASET_PATH)
    print(dataset)

    # 2. Load model & tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit_runtime = USE_4BIT and bitsandbytes_available()
    print(f"MODEL_NAME={MODEL_NAME}")
    print(f"USE_4BIT_REQUESTED={USE_4BIT}")
    print(f"USE_4BIT_RUNTIME={use_4bit_runtime}")

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if use_4bit_runtime:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )
    model.config.use_cache = False

    if use_4bit_runtime:
        model = prepare_model_for_kbit_training(model)

    # 3. Wrap with LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    # 5. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        args=training_args,
    )

    # 6. Train
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
