# mnist_model_v4.py
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
# ---------------------------
# Model: C C M  C C M  C C P
# ---------------------------

class CCMCCMCCP(nn.Module):
    """Three CC blocks with two MaxPools and Global AvgPool head (GAP)."""
    def __init__(self, ch=(8, 12, 16, 20, 24, 24)):
        super().__init__()
        c1, c2, c3, c4, c5, c6 = ch

        # Block 1: C C M (28x28 -> 14x14)
        self.c1 = nn.Conv2d(1,  c1, kernel_size=5, padding=2, bias=False)
        self.b1 = nn.BatchNorm2d(c1)
        self.c2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c2)
        self.m1 = nn.MaxPool2d(2)

        # Block 2: C C M (14x14 -> 7x7)
        self.c3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(c3)
        self.c4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(c4)
        self.m2 = nn.MaxPool2d(2)

        # Block 3: C C P (7x7 -> 1x1 via GAP)
        self.c5 = nn.Conv2d(c4, c5, kernel_size=3, padding=1, bias=False)
        self.b5 = nn.BatchNorm2d(c5)
        self.c6 = nn.Conv2d(c5, c6, kernel_size=3, padding=1, bias=False)
        self.b6 = nn.BatchNorm2d(c6)

        self.fc = nn.Linear(c6, 10)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = self.m1(x)

        x = F.relu(self.b3(self.c3(x)))
        x = F.relu(self.b4(self.c4(x)))
        x = self.m2(x)

        x = F.relu(self.b5(self.c5(x)))
        x = F.relu(self.b6(self.c6(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # GAP
        return self.fc(x)  # logits



# ---------------------------
# Data / Config
# ---------------------------

@dataclass
class Config:
    epochs: int = 15
    batch_size: int = 128
    test_batch_size: int = 512
    lr: float = 0.08
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    seed: int = 1
    channels: tuple = (8, 12, 16, 20, 24, 24)  # <~16k params
    data_dir: str = "./data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_loaders(cfg: Config):
    mean, std = (0.1307,), (0.3081,)
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.MNIST(cfg.data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.test_batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------
# Train / Eval
# ---------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, 100.0 * correct / total


def train(cfg: Config):
    # Repro
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_loaders(cfg)

    model = CCMCCMCCP(cfg.channels).to(cfg.device)
    print(summary(model, (1, 28, 28)))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Per-step warmup (1 epoch) + cosine
    steps_per_epoch = len(train_loader)
    warmup_steps = steps_per_epoch * 1
    total_steps = steps_per_epoch * cfg.epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    global_step = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1



        tl, ta = evaluate(
            model, test_loader, cfg.device, criterion
        )

        trl, tra= evaluate(
            model, train_loader, cfg.device, criterion
        )


        print(f"Epoch {ep:02d} | test loss {tl:.4f} | test acc {ta:.2f}%| train acc{tra:.2f}%| train loss{trl:.2f}%")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")


    torch.save(model.state_dict(), "mnist_ccmccmccp_ema.pt")


    return model

if __name__ == '__main__':
    train(Config)