import os

# ==========================================
# 0. FORÇAR O DIRETÓRIO DE CACHE GLOBAL
# ==========================================
# Define o caminho exato onde TUDO será salvo
CACHE_DIR = r"C:\Users\Igor Henrique\OneDrive\Documents\AromaLLM\cache"

# Força o Hugging Face a usar APENAS essa pasta, alterando as variáveis do sistema
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# Cria a pasta caso ela ainda não exista
os.makedirs(CACHE_DIR, exist_ok=True)

import torch
import pandas as pd
import numpy as np
import re
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "OpenDFM/ChemDFM-v1.5-8B"
OUTPUT_DIR = "./AromaLLM_finetuned"
DATA_FILE = "combined_odor_prompts.csv"

# ==========================================
# 2. DATA PREPARATION (Train/Test Split)
# ==========================================
print("Loading and splitting dataset...")
df = pd.read_csv(DATA_FILE)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print(f"Training samples: {len(train_df)} | Testing samples: {len(test_df)}")

# ==========================================
# 3. MODEL & TOKENIZER SETUP (QLoRA + CPU Offload)
# ==========================================
print(f"Loading tokenizer and model: {MODEL_ID} in 4-bit with CPU offloading...")
print(f"All files will be saved in and loaded from: {CACHE_DIR}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configuração 4-bit COM permissão para usar a memória RAM do PC
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True # Resolve erro de VRAM
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 4. TRAINING
# ==========================================
print("Starting training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, # Reduzido para 1 para salvar VRAM
    gradient_accumulation_steps=8, 
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=3,
    max_steps=-1,
    fp16=True,
    group_by_length=True,
    gradient_checkpointing=True # Ativado para economizar muita VRAM
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="chemdfm_text",
    max_seq_length=512,
    args=training_args,
    peft_config=lora_config,
)

trainer.train()

trainer.model.save_pretrained(f"{OUTPUT_DIR}/AromaLLM-Adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/AromaLLM-Adapter")
print(f"Model saved to {OUTPUT_DIR}/AromaLLM-Adapter")

# ==========================================
# 5. EVALUATION (Inference on Test Set)
# ==========================================
print("Starting Evaluation on Test Set...")
model.eval()

predictions = []
actual_primaries = []
actual_subs = []
predicted_primaries = []
predicted_subs = []

def parse_odors(text):
    prim_match = re.search(r'primary odors present in this molecule are:\s*(.*?)\.', text)
    sub_match = re.search(r'The sub-odors are:\s*(.*?)\.', text)
    
    prims = [o.strip() for o in prim_match.group(1).split(',')] if prim_match else []
    subs = [o.strip() for o in sub_match.group(1).split(',')] if sub_match else []
    
    prims = [p for p in prims if p and p != "None"]
    subs = [s for s in subs if s and s != "None"]
    return prims, subs

for idx, row in test_df.iterrows():
    input_text = f"[Round 0]\nHuman: {row['prompt']}\nAssistant:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):].strip()
    predictions.append(generated_text)
    
    true_prims, true_subs = parse_odors(row['completion'])
    actual_primaries.append(true_prims)
    actual_subs.append(true_subs)
    
    pred_prims, pred_subs = parse_odors(generated_text)
    predicted_primaries.append(pred_prims)
    predicted_subs.append(pred_subs)

test_df['generated_text'] = predictions
test_df.to_csv(f"{OUTPUT_DIR}/test_set_predictions.csv", index=False)

# ==========================================
# 6. METRICS CALCULATION
# ==========================================
print("Calculating Metrics...")

def calculate_multilabel_metrics(y_true, y_pred, name):
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true + y_pred) 
    
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    
    return {"Dataset": name, "Precision": precision, "Recall": recall, "F1_Score": f1}

metrics = []
metrics.append(calculate_multilabel_metrics(actual_primaries, predicted_primaries, "Primary Odors"))
metrics.append(calculate_multilabel_metrics(actual_subs, predicted_subs, "Sub Odors"))

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f"{OUTPUT_DIR}/evaluation_metrics.csv", index=False)

print("\n--- FINAL EVALUATION METRICS ---")
print(metrics_df.to_string(index=False))
print(f"\nTraining and Evaluation complete! All files saved in {OUTPUT_DIR}")