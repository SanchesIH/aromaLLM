import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Dirs ──────────────────────────────────────────────────────────────────────
base_dir       = "/home/henrich/Documents/sofiax"
model_save_dir = os.path.join(base_dir, "chemdfm_finetuned")

# ── Load Tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_save_dir, trust_remote_code=True)

# ── Setup 4-bit & BF16 (same as training) ─────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ── Load Base Model ───────────────────────────────────────────────────────────
base_model_id = "OpenDFM/ChemDFM-v1.5-8B"
print(f"Loading base model ({base_model_id})...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# ── Merge Fine-Tuned Weights (LoRA) ───────────────────────────────────────────
print("Applying your fine-tuned weights...")
model = PeftModel.from_pretrained(base_model, model_save_dir)
model.eval()
print("Model ready!\n")

# ── Helper to format the prompt ───────────────────────────────────────────────
def generate_response(prompt_text):
    # Format identically to how it was trained, but leave Assistant blank
    formatted_prompt = f"[Round 0]\nHuman: {prompt_text}\nAssistant:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,   # <--- ADDED: Tells it when to stop
            repetition_penalty=1.15                # <--- ADDED: Punishes the model for repeating words
        )
    
    # Decode only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    
    # <--- ADDED: clean_up_tokenization_spaces=False to fix the warning
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
    
    return response.strip()

# ── Test it out! ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Type 'quit' to exit.")
    while True:
        user_input = input("\nEnter a SMILES string or prompt: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        print("\nThinking...")
        answer = generate_response(user_input)
        print(f"Assistant: {answer}")