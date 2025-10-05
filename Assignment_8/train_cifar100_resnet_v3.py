#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a CIFAR-100 classifier from scratch (no pretrained) using ResNet variants
with optional dilation and OneCycleLR. Designed to hit >=73% top-1 with a strong
but standard recipe (augmentations, label smoothing, SGD+momentum, EMA, OneCycle).

Usage examples:

# ResNet-18 baseline, no dilation
python train_cifar100_resnet.py --arch resnet18 --epochs 120 --batch-size 256 --workers 4 \
    --max-lr 0.4 --weight-decay 5e-4 --label-smoothing 0.1 --ema 0.999 --autoaugment \
    --cutmix-prob 0.5

# ResNet-34 with dilation on later stages
python train_cifar100_resnet.py --arch resnet34 --replace-stride-with-dilation 0 1 1 \
    --epochs 160 --batch-size 256 --max-lr 0.35 --weight-decay 3e-4 --label-smoothing 0.1 \
    --autoaugment --cutmix-prob 0.3

Notes:
- Uses CIFAR-specific ResNet stem (3x3 conv, no initial maxpool).
- Uses OneCycleLR with SGD+Nesterov.
- Optional: AutoAugment/RandAugment, RandomErasing, CutMix/MixUp, EMA.
- Mixed precision (AMP) is enabled by default.

Tested with PyTorch >=2.2 and torchvision >=0.17. Adjust as needed.
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as T

# ------------------------------- Utils ----------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False  # faster with benchmarking
    cudnn.benchmark = True

@dataclass
class AccMeter:
    correct: float = 0.0
    total: float = 0.0
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            self.correct += (pred == targets).float().sum().item()
            self.total += float(targets.numel())
    def update_mixed(self, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
        # Weighted accuracy for MixUp/CutMix
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            self.correct += lam * (pred == y_a).float().sum().item() + (1.0 - lam) * (pred == y_b).float().sum().item()
            self.total += float(y_a.numel())
    def value(self) -> float:
        return 100.0 * self.correct / max(1.0, self.total)

# ------------------------------- Data -----------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def build_transforms(autoaugment: bool, randaugment: bool, random_erasing: float):
    train_trans = [
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
    ]
    if autoaugment:
        train_trans.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    if randaugment:
        train_trans.append(T.RandAugment(num_ops=2, magnitude=9))
    train_trans.extend([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    if random_erasing > 0:
        train_trans.append(T.RandomErasing(p=random_erasing, scale=(0.02, 0.2), ratio=(0.3, 3.3)))

    test_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    return T.Compose(train_trans), test_trans


def get_dataloaders(data_dir: str, batch_size: int, workers: int, autoaugment: bool,
                    randaugment: bool, random_erasing: float):
    train_tf, test_tf = build_transforms(autoaugment, randaugment, random_erasing)
    train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True, drop_last=True, persistent_workers=workers>0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                              pin_memory=True, persistent_workers=workers>0)
    return train_loader, test_loader

# ------------------------------- Model ----------------------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        # In CIFAR, keep 1x1-3x3-1x1 but adapt padding for dilation in the 3x3
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers: List[int], num_classes=100,
                 replace_stride_with_dilation=(False, False, False)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR stem
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # no maxpool for CIFAR

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, dilate=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = 1
        if dilate:
            dilation = 2
            stride = 1  # replace stride by dilation
        else:
            dilation = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes=100, replace_stride_with_dilation=(False, False, False)):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, replace_stride_with_dilation)

def resnet34(num_classes=100, replace_stride_with_dilation=(False, False, False)):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, replace_stride_with_dilation)

def resnet50(num_classes=100, replace_stride_with_dilation=(False, False, False)):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, replace_stride_with_dilation)

# --------------------------- CutMix / MixUp ------------------------------

def mixup_data(x, y, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha: float = 1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size, device=x.device)
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
    cx = torch.randint(0, w, (1,), device=x.device).item()
    cy = torch.randint(0, h, (1,), device=x.device).item()
    x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w); y2 = min(cy + cut_h // 2, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def criterion_with_smoothing(logits, targets, smoothing=0.0):
    if smoothing <= 0:
        return F.cross_entropy(logits, targets)
    # Label smoothing CE
    n_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# ------------------------------ EMA -------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# --------------------------- Train / Evaluate ----------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, args, epoch, ema: Optional[EMA]):
    model.train()
    acc = AccMeter(); loss_sum = 0.0; steps = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Optional MixUp/CutMix
        if args.cutmix_prob > 0 and random.random() < args.cutmix_prob:
            images, y_a, y_b, lam = cutmix_data(images, targets, alpha=args.cutmix_alpha)
            mix_mode = 'cutmix'
        elif args.mixup_alpha > 0:
            images, y_a, y_b, lam = mixup_data(images, targets, alpha=args.mixup_alpha)
            mix_mode = 'mixup'
        else:
            y_a = y_b = targets; lam = 1.0; mix_mode = 'none'

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=(torch.cuda.is_available() and not args.no_amp)):
            logits = model(images)
            if mix_mode == 'none':
                loss = criterion_with_smoothing(logits, targets, smoothing=args.label_smoothing)
            else:
                loss = lam * criterion_with_smoothing(logits, y_a, smoothing=args.label_smoothing)+(1 - lam) * criterion_with_smoothing(logits, y_b, smoothing=args.label_smoothing)
        scaler.scale(loss).backward()
        if args.clip_grad and args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        loss_sum += loss.item(); steps += 1
        # Accuracy accounting: if we used MixUp/CutMix this step, use weighted acc
        if mix_mode == 'none':
            acc.update(logits.detach(), targets)
        else:
            acc.update_mixed(logits.detach(), y_a, y_b, lam)
    return loss_sum / max(1, steps), acc.value()


def evaluate(model, loader, device, ema: Optional[EMA] = None):
    model.eval(); acc = AccMeter(); loss_sum = 0.0; steps = 0
    context = torch.no_grad()
    context.__enter__()
    try:
        if ema is not None:
            ema.apply_shadow(model)
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss_sum += loss.item(); steps += 1
            acc.update(logits, targets)
    finally:
        if ema is not None:
            ema.restore(model)
        context.__exit__(None, None, None)
    return loss_sum / max(1, steps), acc.value()

# ------------------------------- Main -----------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-100 ResNet from scratch")
    p.add_argument('--data', type=str, default='./data', help='data directory')
    p.add_argument('--arch', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50'])
    p.add_argument('--replace-stride-with-dilation', type=int, nargs=3, default=[0,0,0],
                   help='Replace stride with dilation for layer2/3/4 (0/1 flags)')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-lr', type=float, default=0.4, help='OneCycle max LR')
    p.add_argument('--pct-start', type=float, default=0.3, help='OneCycle pct_start')
    p.add_argument('--div-factor', type=float, default=25.0)
    p.add_argument('--final-div-factor', type=float, default=1e4)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--nesterov', action='store_true')
    p.add_argument('--clip-grad', type=float, default=0.0, help='clip global grad norm (0 to disable)')
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--mixup-alpha', type=float, default=0.0)
    p.add_argument('--cutmix-prob', type=float, default=0.0)
    p.add_argument('--cutmix-alpha', type=float, default=1.0)
    p.add_argument('--autoaugment', action='store_true')
    p.add_argument('--randaugment', action='store_true')
    p.add_argument('--random-erasing', type=float, default=0.2)
    p.add_argument('--ema', type=float, default=0.0, help='EMA decay, e.g., 0.999 to enable')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--save', type=str, default='./checkpoints')
    return p.parse_args()


def build_model(name: str, dilation_flags: List[int]):
    flags = tuple(bool(x) for x in dilation_flags)
    if name == 'resnet18':
        model = resnet18(num_classes=100, replace_stride_with_dilation=flags)
    elif name == 'resnet34':
        model = resnet34(num_classes=100, replace_stride_with_dilation=flags)
    elif name == 'resnet50':
        model = resnet50(num_classes=100, replace_stride_with_dilation=flags)
    else:
        raise ValueError(name)
    return model


def main():
    args = parse_args()
    os.makedirs(args.save, exist_ok=True)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        autoaugment=args.autoaugment,
        randaugment=args.randaugment,
        random_erasing=args.random_erasing,
    )

    model = build_model(args.arch, args.replace_stride_with_dilation).to(device)

    # Optimizer + OneCycleLR
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.max_lr, momentum=args.momentum,
        nesterov=args.nesterov, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=total_steps,
        pct_start=args.pct_start, anneal_strategy='cos',
        div_factor=args.div_factor, final_div_factor=args.final_div_factor
    )

    scaler = GradScaler(enabled=(torch.cuda.is_available() and not args.no_amp))
    ema = EMA(model, decay=args.ema) if args.ema and args.ema > 0 else None

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, args, epoch, ema)
        val_loss, val_acc = evaluate(model, test_loader, device, ema)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc,
            'args': vars(args),
        }
        torch.save(state, os.path.join(args.save, 'last.pt'))
        if is_best:
            torch.save(state, os.path.join(args.save, 'best.pt'))

        print(f"Epoch {epoch:03d}/{args.epochs} | train: loss {train_loss:.3f} acc {train_acc:.2f}% | "
              f"val: loss {val_loss:.3f} acc {val_acc:.2f}% | best: {best_acc:.2f}%")

    print(f"Training complete. Best top-1: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
