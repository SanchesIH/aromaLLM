import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from rdkit import Chem
from huggingface_hub import login

# ── Critical VRAM fixes ───────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Auth ──────────────────────────────────────────────────────────────────────
login(token=os.environ.get("HF_TOKEN", "YOUR_HF_KEY"))

# ── Dirs ──────────────────────────────────────────────────────────────────────
base_dir       = "/home/henrich/Documents/sofiax"
data_dir       = os.path.join(base_dir, "data")
fig_dir        = os.path.join(base_dir, "figs")
model_save_dir = os.path.join(base_dir, "chemdfm_finetuned")
metrics_dir    = os.path.join(base_dir, "metrics")

for d in [data_dir, fig_dir, model_save_dir, metrics_dir]:
    os.makedirs(d, exist_ok=True)

# ── GPU check ─────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "No CUDA GPU detected!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── SMILES canonicalization ───────────────────────────────────────────────────
def canonicalize_smiles(smiles):
    try:
        return Chem.CanonSmiles(smiles, useChiral=True)
    except:
        return smiles

# ── ChemDFM native prompt format ─────────────────────────────────────────────
def format_chemdfm(prompt, completion, round_idx=0):
    return f"[Round {round_idx}]\nHuman: {prompt}\nAssistant: {completion}"

# ── Load & format data ────────────────────────────────────────────────────────
print("\nLoading and processing dataset...")
df = pd.read_csv(os.path.join(data_dir, "combined_odor_prompts.csv"))

tqdm.pandas(desc="Canonicalizing SMILES")
if 'smiles' in df.columns:
    df['smiles'] = df['smiles'].progress_apply(canonicalize_smiles)

formatted_texts = []
print("Formatting prompts using ChemDFM standard...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Formatting"):
    prompt     = row.get('prompt', '')
    completion = row.get('completion', '')
    formatted_texts.append(format_chemdfm(prompt, completion))

df['chemdfm_text'] = formatted_texts

# ── Split ─────────────────────────────────────────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv(os.path.join(data_dir, "train.csv"),      index=False)
val_df.to_csv(  os.path.join(data_dir, "validation.csv"), index=False)
test_df.to_csv( os.path.join(data_dir, "test.csv"),       index=False)

print(f"  Train:      {len(train_df)} samples")
print(f"  Validation: {len(val_df)}  samples")
print(f"  Test:       {len(test_df)}  samples")

train_dataset = Dataset.from_pandas(train_df[['chemdfm_text']])
val_dataset   = Dataset.from_pandas(val_df[['chemdfm_text']])
test_dataset  = Dataset.from_pandas(test_df[['chemdfm_text']])

# ── Model & tokenizer ─────────────────────────────────────────────────────────
model_id = "OpenDFM/ChemDFM-v1.5-8B"
print(f"\nLoading {model_id} with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # SWITCHED TO BFLOAT16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=False
)
if getattr(tokenizer, "pad_token_id", None) is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # SWITCHED TO BFLOAT16
)

# ── Manually prepare model ────────────────────────────────────────────────────
model.config.use_cache = False

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "norm" in name.lower():
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
torch.cuda.empty_cache()

# ── LoRA ──────────────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
torch.cuda.empty_cache()

# ── Metrics callback ──────────────────────────────────────────────────────────
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses   = []
        self.val_losses     = []
        self.train_steps    = []
        self.val_steps      = []
        self.learning_rates = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(step)
            self.learning_rates.append(logs.get("learning_rate", 0))
            print(f"  [step {step:>5}] train_loss={logs['loss']:.4f}  "
                  f"lr={logs.get('learning_rate', 0):.2e}")
        if "eval_loss" in logs:
            self.val_losses.append(logs["eval_loss"])
            self.val_steps.append(step)
            print(f"  [step {step:>5}] val_loss={logs['eval_loss']:.4f}")

metrics_callback = MetricsCallback()

# ── Training args ─────────────────────────────────────────────────────────────
training_args = SFTConfig(
    dataset_text_field="chemdfm_text",
    max_length=256,
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    bf16=True,  # SWITCHED TO BF16
    max_grad_norm=0.3,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    report_to="none"
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[metrics_callback]
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training...")
trainer.train()
torch.cuda.empty_cache()

# ── Save model ────────────────────────────────────────────────────────────────
print("\nSaving final model...")
trainer.model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
print(f"Model saved to {model_save_dir}")