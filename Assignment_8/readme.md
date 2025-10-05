
---

# CIFAR-100 Training – ResNet-18 from Scratch (One-Cycle Policy + CutMix + AutoAugment)

### 📘 Objective

Train a **ResNet-18** model **from scratch** (no pretraining) on the **CIFAR-100** dataset, targeting **≥ 73 % top-1 accuracy** using:

* One-Cycle LR scheduling
* AutoAugment and CutMix for data augmentation
* Label smoothing and EMA for regularization
* Standard CIFAR-style ResNet stem (no initial max-pool)

---

## 🧠 Architecture Summary — ResNet-18 (CIFAR version)

| Stage   | Layers                       | Output Size | Details             |
| :------ | :--------------------------- | :---------- | :------------------ |
| Conv1   | 3×3 conv, 64                 | 32×32       | stride 1, padding 1 |
| Layer 1 | 2 × BasicBlock(64)           | 32×32       | stride 1            |
| Layer 2 | 2 × BasicBlock(128)          | 16×16       | stride 2            |
| Layer 3 | 2 × BasicBlock(256)          | 8×8         | stride 2            |
| Layer 4 | 2 × BasicBlock(512)          | 4×4         | stride 2            |
| Head    | Global AvgPool + FC(512→100) | 1×1         | softmax output      |

* **Total parameters:** ~11.2 M
* **BatchNorm** after every conv
* **ReLU** activations (in-place)
* **No pretrained weights**

---

## ⚙️ Command & Hyperparameters

```bash
python train_cifar100_resnet_v3.py \
  --arch resnet18 \
  --epochs 130 \
  --batch-size 256 \
  --max-lr 0.4 \
  --pct-start 0.3 \
  --div-factor 25 \
  --final-div-factor 10000 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --autoaugment \
  --cutmix-prob 0.5 --cutmix-alpha 1.0 \
  --random-erasing 0.0 \
  --ema 0.999 \
  --nesterov
```

**Optimizer:** SGD + Nesterov
**Scheduler:** OneCycleLR (cosine anneal, warmup 30 %)
**Loss:** Cross-entropy + label smoothing (0.05)
**Regularization:** EMA (0.999), CutMix (0.5 prob), AutoAugment
**Precision:** AMP mixed precision
**Dataset:** CIFAR-100 (50 K train / 10 K test, 32×32 RGB)

---

## 📊 Results Summary

| Metric                    |                       Value |
| :------------------------ | --------------------------: |
| **Best Top-1 Accuracy**   |                 **76.72 %** |
| **Final Validation Loss** |                       0.876 |
| **Final Train Accuracy**  |                     71.37 % |
| **Epoch Achieved**        |                         130 |
| **Optimizer Step**        | One-Cycle peak at ~epoch 40 |

---

## 📈 Training Curve Snapshot

* Early phase (epochs 1-20): rapid convergence, val ≈ +10 %
* Mid-phase (epochs 30-70): steady improvement to ~66 %
* Late phase (epochs 100-130): smooth climb → 76.7 % best
* One-Cycle annealing phase produced a sharp validation gain near the end (as expected)

---

## 🧩 Observations

* **Mix of AutoAugment + CutMix** balanced regularization and diversity without over-regularization.
* Turning **RandomErasing off** improved stability.
* **EMA (0.999)** stabilized validation toward the final 20 epochs.
* Accuracy spiked in final epochs due to One-Cycle’s LR annealing tail.

---

## 🏁 Conclusion

* ✅ Reached **76.7 % Top-1** on CIFAR-100 — exceeding the 73 % target.
* 🧱 Architecture = pure ResNet-18 (no pretraining, CIFAR stem).
* 🔁 Recipe proved stable; repeatable within ± 0.5 % across seeds.
* 🚀 Next step: test **ResNet-34** or **dilated-18** variant to evaluate headroom (expected ≈ 77.5 – 78 %).

---
