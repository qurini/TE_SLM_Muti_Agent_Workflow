
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   
DATASET_PATH = "hf_data/telegraph_ft_nl2te"
OUTPUT_DIR = "outputs/qwen_te_translator_0_5b_lora"

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

    # 2. Load model & tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto"
    )

    # 3. Wrap with LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # adjust for Qwen if needed
    )
    model = get_peft_model(model, lora_config)

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-4,
        logging_steps=50,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        report_to="none",
    )

    # 5. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=1024,
        args=training_args,
    )

    # 6. Train
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
