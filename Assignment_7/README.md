# CIFAR-10 Custom CNN

## Model Architecture
- Four convolutional blocks: C1 → C2 → C3 → C4
- No MaxPooling layers (requirement).
- C2 and C3 use **dilated kernels** (dilation=2,4).
- All convolutions are **depthwise separable**.
- C4 uses **stride=2** to reduce spatial dimension.
- Global Average Pooling (GAP) + Fully Connected layer (256 → 10).

**Receptive Field**: Final RF ≈ 47 × 47 (> 44 required).  
**Total Parameters**: 184,824 (well below 200k).

---

## Data Augmentation (Albumentations)
Applied to training set:
- `HorizontalFlip(p=0.5)`
- `ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15)`
- `CoarseDropout(max_holes=1, min_holes=1, max_height=16, min_height=16, max_width=16, min_width=16, fill_value=mean_255, mask_fill_value=None)`  
- `Normalize(mean, std)`
- `ToTensorV2()`

Applied to test set:
- `Normalize(mean, std)`
- `ToTensorV2()`

---

## Training Details
- Optimizer: **Adam**, `lr=0.001`, `weight_decay=5e-4`
- Loss: **CrossEntropyLoss(label_smoothing=0.05)**
- Scheduler: **OneCycleLR** (`max_lr=0.01`, epochs=25)
- Mixed Precision: **Yes (torch.cuda.amp)**
- Batch Size: 128
- Epochs: 25
- Seed: 42 (for reproducibility)

---

## Results
- **Best Test Accuracy**: **86.91%** (epoch 25):contentReference[oaicite:1]{index=1}
- **Training Accuracy**: 88.33%
- Stable convergence after ~20 epochs.

---

## Logs (Highlights)

[Log](./logs.txt)  