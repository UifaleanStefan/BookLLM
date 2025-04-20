import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === CONFIG ===
model_id = "meta-llama/Llama-2-7b-hf"
dataset_path = "/path/on/hpc/processed_books_tokenized_final.jsonl"  # <-- update this on HPC
output_dir = "/path/on/hpc/output/llama-7b-finetuned"                # <-- update this on HPC

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # Required for Causal LM

# === Load dataset ===
dataset = load_dataset("json", data_files=dataset_path, split="train")

# === Tokenization ===
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# === BitsAndBytes QLoRA config ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",  # Change to "float16" if bfloat16 not supported
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# === Load base model with multi-GPU support ===
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # ✅ Spread across multiple GPUs
    use_auth_token=True,
    trust_remote_code=True
)

# === LoRA Fine-tuning setup ===
base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

# === Training configuration (multi-GPU aware) ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,                   # Per GPU
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=True,
    report_to="none",
    save_total_limit=2,
    ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None  # ✅ Required for multi-GPU
)

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Train ===
trainer.train()

# === Save ===
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
