import torch
import os

def save_checkpoint(model, optimizer, step, loss, path):
    checkpoint = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(path, model, optimizer):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
        
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from step {step}")
    return step, loss 