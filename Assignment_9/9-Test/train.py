import os
import math
import time
import json
import shutil
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# --------------------
# Logging
# --------------------
logger = logging.getLogger("imagenet_train")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)


# --------------------
# Utils
# --------------------
def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def mixup_data(x, y, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Simple Mixup (no CutMix)"""
    if alpha <= 0.0:
        return x, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, torch.stack([y_a, y_b], dim=1), lam


def mixup_criterion(criterion, pred, target_mix, lam):
    if lam == 1.0 or target_mix.ndim == 1:
        return criterion(pred, target_mix)
    y_a, y_b = target_mix[:, 0], target_mix[:, 1]
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WarmupCosine:
    """Warmup then Cosine Annealing over epochs."""
    def __init__(self, optimizer, base_lr, min_lr, warmup_epochs, total_epochs, iters_per_epoch):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch
        self.t = 0  # global step

    def step(self):
        self.t += 1
        epoch_progress = self.t / max(1, self.iters_per_epoch)
        current_epoch = epoch_progress  # float-like epoch

        if current_epoch < self.warmup_epochs:
            lr = self.base_lr * current_epoch / max(1e-8, self.warmup_epochs)
        else:
            # cosine over remaining
            progress = (current_epoch - self.warmup_epochs) / max(1e-8, (self.total_epochs - self.warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# --------------------
# Trainer
# --------------------
class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # Model
        # was: self.model = resnet50(weights=None, num_classes=1000)
        self.model = resnet50(weights=None, num_classes=self.cfg["num_classes"])

        logger.info("Initializing ResNet-50 from scratch (weights=None)")
        logger.info(f"Num classes = {self.cfg['num_classes']}")

        self.model.to(self.device)

        # Data
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg["label_smoothing"]).to(self.device)

        # Optimizer
        lr = self.cfg["base_lr"] * (self.cfg["batch_size"] / 256.0)
        if self.cfg["optimizer"].lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.05)
            logger.info("Using AdamW (lr=3e-4, wd=0.05)")
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.cfg["momentum"],
                weight_decay=self.cfg["weight_decay"],
                nesterov=True,
            )
            logger.info(f"Using SGD (lr={lr:.5f}, momentum={self.cfg['momentum']}, wd={self.cfg['weight_decay']})")

        # LR scheduler: warmup + cosine
        self.lr_sched = WarmupCosine(
            optimizer=self.optimizer,
            base_lr=lr,
            min_lr=self.cfg["cosine_min_lr"],
            warmup_epochs=self.cfg["warmup_epochs"],
            total_epochs=self.cfg["epochs"],
            iters_per_epoch=len(self.train_loader),
        )

        # AMP
        self.scaler = GradScaler(enabled=self.cfg["use_amp"])

        # SWA (set up but activate only in the last N epochs)
        self.use_swa = bool(self.cfg.get("use_swa", True))
        self.swa_last_epochs = int(self.cfg.get("swa_last_epochs", 10))
        self.swa_model = AveragedModel(self.model) if self.use_swa else None
        self.swa_scheduler = None  # will be created when SWA starts

        # Misc
        self.best_acc1 = 0.0
        self.epoch = 0

        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    def _randaugment_if_available(self):
        # Torchvision >= 0.13 has RandAugment
        try:
            from torchvision.transforms import RandAugment
            return RandAugment(num_ops=2, magnitude=9)
        except Exception:
            logger.warning("RandAugment not available in this torchvision. Continuing without it.")
            return None

    def _build_dataloaders(self):
        data_dir = Path(self.cfg["data_dir"])
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"

        train_tfms = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]

        if self.cfg.get("use_randaugment", True):
            ra = self._randaugment_if_available()
            if ra is not None:
                train_tfms.append(ra)

        train_tfms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if self.cfg.get("random_erasing_p", 0.0) > 0:
            # RandomErasing works on tensors after normalization
            train_tfms.append(transforms.RandomErasing(p=self.cfg["random_erasing_p"]))

        val_tfms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        train_set = ImageFolder(str(train_dir), transform=transforms.Compose(train_tfms))
        val_set = ImageFolder(str(val_dir), transform=val_tfms)

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=self.cfg["pin_memory"],
            prefetch_factor=self.cfg["prefetch_factor"],
            drop_last=True,
            persistent_workers=True if self.cfg["num_workers"] > 0 else False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=self.cfg["pin_memory"],
            prefetch_factor=self.cfg["prefetch_factor"],
            drop_last=False,
            persistent_workers=True if self.cfg["num_workers"] > 0 else False,
        )
        return train_loader, val_loader

    # --------------------
    # Train / Validate
    # --------------------
    def train(self):
        epochs = self.cfg["epochs"]
        start_time = time.time()

        for epoch in range(epochs):
            self.epoch = epoch
            train_stats = self._train_one_epoch()
            val_stats = self._validate()

            # Save best
            is_best = val_stats["acc1"] > self.best_acc1
            if is_best:
                self.best_acc1 = val_stats["acc1"]
                self._save_checkpoint("model_best.pth", is_best=True, extra=val_stats)

            # Periodic checkpoint
            if (epoch + 1) % self.cfg["checkpoint_frequency"] == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", is_best=False, extra=val_stats)

            logger.info(f"Epoch {epoch+1}/{epochs} | "
                        f"Train loss {train_stats['loss']:.4f} | "
                        f"Val acc@1 {val_stats['acc1']:.2f} | acc@5 {val_stats['acc5']:.2f} | "
                        f"LR {self.lr_sched.get_lr():.6f}")

            # Early stop (usually keep off for from-scratch)
            if self.cfg.get("early_stopping_patience", 0) > 0:
                # (Left as no-op hook; recommend training full schedule)
                pass

        # If SWA used, update BN and save final SWA weights
        if self.use_swa and self.swa_model is not None:
            logger.info("Updating BN stats for SWA model...")
            update_bn(self.train_loader, self.swa_model, device=self.device)
            self._save_state_dict(self.swa_model.module.state_dict(), "model_swa.pth")

        elapsed = (time.time() - start_time) / 3600.0
        logger.info(f"Finished training. Best acc@1: {self.best_acc1:.2f}. Total time: {elapsed:.2f} h.")

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        n = 0

        # Activate SWA if we are in the last N epochs (once)
        if self.use_swa and self.epoch == (self.cfg["epochs"] - self.swa_last_epochs):
            logger.info(f"Activating SWA for last {self.swa_last_epochs} epochs.")
            self.swa_model = AveragedModel(self.model).to(self.device)
            self.swa_scheduler = SWALR(
                self.optimizer,
                anneal_strategy="cos",
                anneal_epochs=self.cfg.get("swa_anneal_epochs", 1),
                swa_lr=self.lr_sched.get_lr() * self.cfg.get("swa_lr_mult", 0.5),
            )

        scaler = self.scaler
        accum = max(1, int(self.cfg.get("gradient_accumulation_steps", 1)))
        use_mix = bool(self.cfg.get("use_mixup", True))
        mix_alpha = float(self.cfg.get("mixup_alpha", 0.2))

        for i, (images, target) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # LR step per iteration
            self.lr_sched.step()

            if use_mix:
                images, target_mix, lam = mixup_data(images, target, alpha=mix_alpha)
            else:
                target_mix, lam = target, 1.0

            with autocast(enabled=self.cfg["use_amp"]):
                output = self.model(images)
                loss = mixup_criterion(self.criterion, output, target_mix, lam) / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                if self.cfg["grad_clip_value"] is not None and self.cfg["grad_clip_value"] > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip_value"])

                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # SWA LR schedule only if SWA is active now
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()

            running_loss += loss.item() * accum
            n += images.size(0)

            if (i + 1) % self.cfg.get("log_interval", 50) == 0:
                logger.info(f"Iter {i+1}/{len(self.train_loader)} | "
                            f"loss {running_loss / n:.4f} | lr {self.lr_sched.get_lr():.6f}")

        # SWA model averaging step each epoch after activation
        if self.swa_model is not None:
            self.swa_model.update_parameters(self.model)

        return {"loss": running_loss / n}

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        top1, top5, total = 0.0, 0.0, 0
        loss_sum = 0.0

        for images, target in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg["use_amp"]):
                output = self.model(images)
                loss = self.criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            bsz = images.size(0)
            loss_sum += loss.item() * bsz
            top1 += acc1 * bsz
            top5 += acc5 * bsz
            total += bsz

        return {"loss": loss_sum / total, "acc1": top1 / total, "acc5": top5 / total}

    # --------------------
    # Checkpointing
    # --------------------
    def _save_checkpoint(self, filename: str, is_best: bool, extra: dict):
        state = {
            "epoch": self.epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_acc1": self.best_acc1,
            "cfg": self.cfg,
            "extra": extra,
        }
        path = Path(self.cfg["checkpoint_dir"]) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path))
        if is_best:
            shutil.copy(str(path), str(Path(self.cfg["checkpoint_dir"]) / "model_best_copy.pth"))

    def _save_state_dict(self, state_dict, filename: str):
        path = Path(self.cfg["checkpoint_dir"]) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state_dict}, str(path))


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    from config import config as cfg
    trainer = Trainer(cfg)
    trainer.train()
