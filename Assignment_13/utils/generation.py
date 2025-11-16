import torch

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits from model
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Sample from logits
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append token and continue
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we hit the end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0]) 