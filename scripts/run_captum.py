import os

# Force bitsandbytes to use its CUDA 13.0 binary on CUDA 13.2 systems
os.environ["BNB_CUDA_VERSION"] = "130"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from captum.attr import LayerIntegratedGradients

# Directories
base_dir       = r"C:\Users\igorh\OneDrive\Documents\SOFIA\aromaLLM"
model_save_dir = os.path.join(base_dir, "chemdfm_finetuned")

# Clear CUDA cache before loading
torch.cuda.empty_cache()

# Load Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_save_dir, trust_remote_code=True)

# Setup 4-bit & BF16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Cap GPU memory allocation to fit comfortably within 8 GB VRAM
max_memory = {0: "6.5GiB", "cpu": "24GiB"}

# Load Base Model
base_model_id = "OpenDFM/ChemDFM-v1.5-8B"
print(f"Loading base model ({base_model_id})...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Merge Fine-Tuned Weights (LoRA)
print("Applying fine-tuned weights...")
model = PeftModel.from_pretrained(base_model, model_save_dir)
model.eval()
print("Model ready!\n")

# Forward Pass Wrapper that extracts raw logits Tensor for Captum
def forward_func(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits[:, -1, :]

# Target the embedding layer in the underlying Llama architecture
embedding_layer = model.base_model.model.model.embed_tokens
lig = LayerIntegratedGradients(forward_func, embedding_layer)

def analyze_smiles_attribution(prompt_text, n_steps=20):
    formatted_prompt = f"[Round 0]\nHuman: {prompt_text}\nAssistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", clean_up_tokenization_spaces=False).to("cuda")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        logits = forward_func(input_ids, attention_mask)
        predicted_token_id = torch.argmax(logits, dim=-1).item()

    predicted_token_str = tokenizer.decode([predicted_token_id])

    attributions, delta = lig.attribute(
        inputs=input_ids,
        additional_forward_args=(attention_mask,),
        target=predicted_token_id,
        n_steps=n_steps,
        return_convergence_delta=True
    )

    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    attributions_normalized = attributions_sum / torch.norm(attributions_sum)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_score_pairs = list(zip(tokens, attributions_normalized.cpu().tolist()))

    return token_score_pairs, predicted_token_str, delta.item()

# Execution Loop
if __name__ == "__main__":
    test_prompt = "Predict the odor profile for this SMILES: CC(=O)OC1=CC=CC=C1C(=O)O"
    
    print(f"\nAnalyzing prompt: {test_prompt}")
    scores, predicted_next_token, convergence_delta = analyze_smiles_attribution(test_prompt)
    
    print(f"\nPredicted target token: '{predicted_next_token}'")
    print(f"Convergence Delta: {convergence_delta:.4f}")
    print("\nToken Attribution Scores:")
    
    for token, score in scores:
        print(f"{token:<20} | Score: {score:+.4f}")