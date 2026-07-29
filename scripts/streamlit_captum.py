def analyze_smiles_attribution(smiles_string, n_steps=20):
    smiles_string = smiles_string.strip()
    
    # 1. Build the full prompt in the background
    prefix = "\nHuman: Output ONLY a comma-separated list of odor descriptors for this SMILES, with no introductory text: "
    suffix = " \nAssistant:"
    formatted_prompt = f"{prefix}{smiles_string}{suffix}"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", clean_up_tokenization_spaces=False).to("cuda")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        # Generate full prediction for text display
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.2, # <--- Fix 1: Penalizes the model for repeating tokens
            pad_token_id=tokenizer.eos_token_id
        )
        new_token_ids = output_ids[0][input_ids.shape[-1]:]
        raw_prediction = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        # Fix 2: Clean up the output string to ensure no duplicates and no intro text
        # If the model stubbornly outputs an intro like "... is: ", split and take the last part
        if ":" in raw_prediction:
            raw_prediction = raw_prediction.split(":", 1)[-1].strip()
            
        # Split by comma, remove duplicates, and ignore empty strings
        items = [item.strip() for item in raw_prediction.split(",")]
        unique_items = []
        for item in items:
            # Remove punctuation (like periods at the end) for clean matching
            clean_item = item.replace(".", "").strip()
            # Only add it if we haven't seen it yet (case-insensitive)
            if clean_item and clean_item.lower() not in [u.lower() for u in unique_items]:
                unique_items.append(clean_item)
                
        full_prediction = ", ".join(unique_items)
        if not full_prediction:
            full_prediction = raw_prediction # Fallback just in case it breaks

        # Get logits for just the very first generated token (for the chart)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        predicted_token_id = torch.argmax(logits, dim=-1).item()

    # Calculate Attributions
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
    scores = attributions_normalized.cpu().tolist()
    clean_tokens = [t.replace('Ġ', ' ').strip() for t in tokens]
    
    # 2. Slice out ONLY the SMILES tokens for the chart
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"][0]
    start_idx = len(prefix_ids)
    
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    end_idx = len(clean_tokens) - len(suffix_ids)
    
    # Fallback to prevent crashes if tokenization is unusual
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = len(clean_tokens)
        
    smiles_tokens = clean_tokens[start_idx:end_idx]
    smiles_scores = scores[start_idx:end_idx]
    
    token_labels = [f"{i}: {t}" for i, t in enumerate(smiles_tokens)]
    
    df = pd.DataFrame({
        "ID": list(range(len(token_labels))),
        "Token": token_labels,
        "Attribution Score": smiles_scores
    })
    
    return df, delta.item(), full_prediction