import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

# ── Directories ───────────────────────────────────────────────────────────────
base_dir       = "/home/henrich/Documents/sofiax"
model_save_dir = os.path.join(base_dir, "chemdfm_finetuned")
fig_dir        = os.path.join(base_dir, "figs")
metrics_dir    = os.path.join(base_dir, "metrics")
test_csv_path  = "/home/henrich/Documents/aromaLLM/data/test.csv" 

os.makedirs(fig_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# ── Load Tokenizer & Model ────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_save_dir, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model_id = "OpenDFM/ChemDFM-v1.5-8B"
print(f"Loading base model ({base_model_id})...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

print("Applying fine-tuned weights...")
model = PeftModel.from_pretrained(base_model, model_save_dir)
model.eval()
print("Model ready!\n")

# ── Generation & Extraction Functions ─────────────────────────────────────────
def generate_response(prompt_text):
    formatted_prompt = f"[Round 0]\nHuman: {prompt_text}\nAssistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
    
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response.strip()

def extract_labels(text):
    if not isinstance(text, str): 
        return []
    primary, sub = [], []
    match_prim = re.search(r'primary odors.*?are:\s*(.*?)(?:\.|$)', text)
    if match_prim:
        primary = [l.strip() for l in match_prim.group(1).split(',') if l.strip()]
    match_sub = re.search(r'sub-odors.*?are:\s*(.*?)(?:\.|$)', text)
    if match_sub:
        sub = [l.strip() for l in match_sub.group(1).split(',') if l.strip()]
    return list(set(primary + sub))

# ── Main Execution ────────────────────────────────────────────────────────────
print(f"Loading test dataset from {test_csv_path}...")
df = pd.read_csv(test_csv_path)

print("Running inference on the test set...")
predictions = []
for prompt in tqdm(df['prompt'], desc="Generating Answers"):
    predictions.append(generate_response(prompt))

df['model_answer'] = predictions

predictions_csv_path = os.path.join(metrics_dir, "test_predictions_full.csv")
df.to_csv(predictions_csv_path, index=False)
print(f"Saved generated predictions to {predictions_csv_path}")

print("Extracting labels and calculating metrics...")
df['true_labels'] = df['completion'].apply(extract_labels)
df['pred_labels'] = df['model_answer'].apply(extract_labels)

mlb = MultiLabelBinarizer()
mlb.fit(df['true_labels'].tolist() + df['pred_labels'].tolist())

y_true = mlb.transform(df['true_labels'])
y_pred = mlb.transform(df['pred_labels'])

metrics = {
    "Micro Precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
    "Macro Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
    "Micro Sensitivity": recall_score(y_true, y_pred, average='micro', zero_division=0),
    "Macro Sensitivity": recall_score(y_true, y_pred, average='macro', zero_division=0),
    "Micro F1": f1_score(y_true, y_pred, average='micro', zero_division=0),
    "Macro F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
    "MCC": matthews_corrcoef(y_true.flatten(), y_pred.flatten())
}

metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
metrics_csv_path = os.path.join(metrics_dir, "evaluation_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved classification metrics to {metrics_csv_path}")

print("Generating performance chart...")
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("ChemDFM Odor Multi-Label Classification Performance", fontsize=16, fontweight='bold')

names = list(metrics.keys())
scores = list(metrics.values())
x_pos = np.arange(len(names))

bars = ax.bar(x_pos, scores, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(names))))

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f'{yval:.4f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
ax.set_ylim(0, max(scores) * 1.20 if scores else 1.0) 
ax.set_ylabel("Score", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
chart_path = os.path.join(fig_dir, "classification_metrics_chart.png")
plt.savefig(chart_path, dpi=200)
print(f"Chart saved successfully to {chart_path}")