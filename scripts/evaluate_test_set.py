import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

# ── Dirs & Paths ──────────────────────────────────────────────────────────────
base_dir = "/home/henrich/Documents/aromaLLM"
model_save_dir = os.path.join(base_dir, "chemdfm_finetuned")
metrics_dir = os.path.join(base_dir, "metrics")
fig_dir = os.path.join(base_dir, "figs")

os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
test_csv_path = "/home/henrich/Documents/aromaLLM/data/test.csv"

# ── 1. Load Tokenizer & LLM ───────────────────────────────────────────────────
print("Loading tokenizer and ChemDFM...")
tokenizer = AutoTokenizer.from_pretrained(model_save_dir, trust_remote_code=True)
tokenizer.padding_side = "left" # Better for generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "OpenDFM/ChemDFM-v1.5-8B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, model_save_dir)
model.eval()

# ── 2. Load STS Embedding Model ───────────────────────────────────────────────
print("Loading lightweight STS model...")
# This small model maps sentences to a 384-dimensional dense vector space
sts_model = SentenceTransformer('all-MiniLM-L6-v2')

# ── Helper Functions for Metrics ──────────────────────────────────────────────
def extract_odors(text):
    """Uses Regex to pull the primary and sub-odors from the formatted text into sets."""
    primary, sub = set(), set()
    # Looks for "primary odors ... are: [tags]. The sub-odors ... are: [tags]"
    match = re.search(r"primary odors.*?are:\s*(.*?)\.\s*(?:The\s*)?sub-odors.*?are:\s*(.*?)(?:\.|$)", str(text), re.IGNORECASE)
    
    if match:
        primary = set([t.strip().lower() for t in match.group(1).split(',') if t.strip()])
        sub = set([t.strip().lower() for t in match.group(2).split(',') if t.strip()])
    return primary, sub

def calculate_f1(true_set, pred_set):
    """Calculates Set-based Precision, Recall, and F1."""
    if not true_set and not pred_set: return 1.0, 1.0, 1.0
    if not true_set or not pred_set: return 0.0, 0.0, 0.0
    
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# ── 3. Generation & Evaluation Loop ───────────────────────────────────────────
df = pd.read_csv(test_csv_path)
results_data = []

print(f"\nStarting generation and evaluation for {len(df)} test molecules...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    smiles = row['smiles']
    prompt_text = row['prompt']
    true_completion = row['completion']
    
    # Format input exactly as trained, leaving Assistant blank
    input_text = f"[Round 0]\nHuman: {prompt_text}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id, 
            repetition_penalty=1.15
        )
    
    # Decode and isolate only the new generated text
    input_length = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    # ─ Calculate Metrics ─
    # 1. Structural Adherence Check
    is_structurally_correct = bool(re.search(r"primary odors.*?are:.*?sub-odors.*?are:", generated_text, re.IGNORECASE))
    
    # 2. Extract Tags and Calculate F1
    true_prim, true_sub = extract_odors(true_completion)
    pred_prim, pred_sub = extract_odors(generated_text)
    
    p_prec, p_rec, p_f1 = calculate_f1(true_prim, pred_prim)
    s_prec, s_rec, s_f1 = calculate_f1(true_sub, pred_sub)
    
    # 3. Semantic Textual Similarity (STS)
    # Cosine similarity between true sentence and predicted sentence
    true_emb = sts_model.encode(true_completion, convert_to_tensor=True)
    pred_emb = sts_model.encode(generated_text, convert_to_tensor=True)
    sts_score = util.cos_sim(true_emb, pred_emb).item()
    
    # Append to results
    results_data.append({
        "smiles": smiles,
        "true_completion": true_completion,
        "generated_text": generated_text,
        "structurally_correct": is_structurally_correct,
        "primary_f1": p_f1,
        "sub_f1": s_f1,
        "sts_score": sts_score
    })

# ── 4. Aggregate & Save ───────────────────────────────────────────────────────
results_df = pd.DataFrame(results_data)

# Save the detailed outputs so you can do the "Eye Test"
detailed_output_path = os.path.join(metrics_dir, "test_generation_results_detailed.csv")
results_df.to_csv(detailed_output_path, index=False)
print(f"\n✓ Saved detailed generation outputs to: {detailed_output_path}")
print("  --> Open this file to manually read and validate the model's responses! (The Eye Test)")

# Calculate Averages
avg_structure = results_df["structurally_correct"].mean()
avg_p_f1 = results_df["primary_f1"].mean()
avg_s_f1 = results_df["sub_f1"].mean()
avg_sts = results_df["sts_score"].mean()

print("\n=== FINAL GENERATION METRICS ===")
print(f"Format Adherence : {avg_structure*100:.1f}%")
print(f"Primary Odors F1 : {avg_p_f1:.4f}")
print(f"Sub-Odors F1     : {avg_s_f1:.4f}")
print(f"Semantic Similarity: {avg_sts:.4f}")

# ── 5. Generate Dashboard Chart ───────────────────────────────────────────────
metrics = ["Format Adherence", "Primary Odor F1", "Sub-Odor F1", "Semantic\nSimilarity"]
scores = [avg_structure, avg_p_f1, avg_s_f1, avg_sts]
colors = ['tab:blue', 'tab:green', 'tab:olive', 'tab:purple']

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, scores, color=colors, width=0.5)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')

plt.title("ChemDFM Generation Task Validation (Unseen Test Set)", fontsize=14, fontweight='bold')
plt.ylabel("Score (0.0 to 1.0)", fontweight='bold')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

chart_path = os.path.join(fig_dir, "generation_metrics_summary.png")
plt.tight_layout()
plt.savefig(chart_path, dpi=200)
print(f"✓ Saved generation metrics chart to: {chart_path}")