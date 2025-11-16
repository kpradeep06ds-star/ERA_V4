import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os
from utils.log_ger import get_logger
logger = get_logger(filename="training.log")

from config.model_config import ModelConfig
from model.smolLM2 import SmolLM2
from training.dataset import TextDataset
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.generation import generate_text

def train():
    # Initialize config and model
    config = ModelConfig()
    model = SmolLM2(config)
    
    # Speed optimizations from class
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=config.use_mixed_precision)
    
    # Load dataset
    dataset = TextDataset("./datasets/input.txt", config.context_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    step = 0
    total_steps = 5000
    checkpoint_path = 'checkpoints/model_5000.pt'
    
    model.train()
    pbar = tqdm(total=total_steps, desc="Training")
    
    while step < total_steps:
        for batch_idx, (x, y) in enumerate(dataloader):
            if step >= total_steps:
                break
                
            t0 = time.time()
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=config.use_mixed_precision):
                logits, loss = model(x, y)
            
            # Backward pass with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate tokens per second
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = config.batch_size * config.context_length / dt
            
            # Log progress
            if step % 500 == 0:
                logger.info(f"\nStep {step} | Loss: {loss.item():.4f} | Tokens/sec: {tokens_per_sec:.2f}")
                # Generate sample text
                sample_text = generate_text(
                    model, 
                    dataset.tokenizer, 
                    "Once upon a time", 
                    max_new_tokens=50
                )
                logger.info(f"Sample text: {sample_text}\n")
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    # Save final checkpoint at step 5000
    save_checkpoint(model, optimizer, step, loss.item(), checkpoint_path)
    logger.info("Completed initial 5000 steps training!")
    
    # Continue training for 50 more steps
    logger.info("\nLoading checkpoint and continuing training...")
    step, prev_loss = load_checkpoint(checkpoint_path, model, optimizer)
    
    total_extra_steps = 50
    pbar = tqdm(total=total_extra_steps, desc="Additional training")
    
    while step < 5050:  # 5000 + 50 additional steps
        for batch_idx, (x, y) in enumerate(dataloader):
            if step >= 5050:
                break
                
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', enabled=config.use_mixed_precision):
                logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                logger.info(f"\nStep {step} | Loss: {loss.item():.4f}")
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Save final model
    save_checkpoint(model, optimizer, step, loss.item(), 'checkpoints/model_final.pt')
    logger.info("Training completed!")

if __name__ == "__main__":
    train() 