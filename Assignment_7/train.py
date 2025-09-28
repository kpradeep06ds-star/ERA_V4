import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torchsummary import summary
import gc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import albumentations_transform, AlbumentationsDataset, CIFAR10Model, mean, std

import random, numpy as np
def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_model():
    set_seed(42)
    # GPU Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    
    # Data Preparation
    train_dataset = CIFAR10(root="../data", train=True, download=True)
    test_dataset = CIFAR10(root="../data", train=False, download=True)

    train_dataset = AlbumentationsDataset(train_dataset, transform=albumentations_transform())
    
    # Create test transform using Albumentations
    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    test_dataset = AlbumentationsDataset(test_dataset, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, 
                           num_workers=4, pin_memory=True)

    model = CIFAR10Model().to(device)

    print(summary(model, input_size=(3, 32, 32)))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=25, 
                          steps_per_epoch=len(train_loader))
    
    # Gradient Scaler for mixed precision training
    scaler = GradScaler()

    # Training Loop
    num_epochs = 25
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100. * correct / total:.2f}%'})

        epoch_loss = train_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()
