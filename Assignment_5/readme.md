# MNIST CNN – CCM-CCM-CC-GAP Architecture

A compact CNN for MNIST classification that achieves **99.54% accuracy** within **20 epochs** and **<20k parameters**.  
This design follows the pattern:

## Table of Contents
- [Architecture](#-architecture)
- [Usage](#-usage)
- [Training Logs](#-training-logs)


Where:
- **C** = Convolution + BatchNorm + ReLU  
- **M** = MaxPool2d  
- **P** = Global Average Pooling  

Final classifier is a single fully-connected `Linear` layer.

---

## 📐 Architecture

Default channel configuration: `(8, 12, 16, 20, 24, 24)`  
Total parameters: **15,634**

```

Input (1x28x28)
├── Conv(1→8, k=5) + BN + ReLU
├── Conv(8→12, k=3) + BN + ReLU
├── MaxPool(2) → 14x14
├── Conv(12→16, k=3) + BN + ReLU
├── Conv(16→20, k=3) + BN + ReLU
├── MaxPool(2) → 7x7
├── Conv(20→24, k=3) + BN + ReLU
├── Conv(24→24, k=3) + BN + ReLU
├── Global Average Pooling (→ 24-dim vector)
└── Linear(24→10)

```


---

## 🚀 Usage

### Training
```bash
# default (20 epochs, batch=128, EMA + TTA enabled)
python mnist_model_v4.py

# wider variant (still <20k params)
python mnist_model_v4.py --channels 10,14,18,24,28,28

# disable EMA or TTA if desired
python mnist_model_v4.py --no-ema --no-tta

Arguments

--epochs : number of epochs (default 20)

--batch-size : training batch size (default 128)

--lr : base learning rate (default 0.08)

--channels : comma-separated channel tuple (default 8,12,16,20,24,24)

--no-ema : disable Exponential Moving Average

--no-tta : disable Test-Time Augmentation

The script saves the final EMA weights as mnist_ccmccmccp_ema.pt
```
---

## Training Logs

**Default Run**

```
Epoch 01 | test loss 2.3139 | test acc 10.11%
Epoch 02 | test loss 2.2978 | test acc  9.80%
Epoch 03 | test loss 2.0077 | test acc 10.56%
Epoch 04 | test loss 1.4377 | test acc 65.90%
Epoch 05 | test loss 1.0169 | test acc 94.03%
Epoch 06 | test loss 0.7830 | test acc 97.45%
Epoch 07 | test loss 0.6369 | test acc 98.32%
Epoch 08 | test loss 0.5376 | test acc 98.73%
Epoch 09 | test loss 0.4581 | test acc 99.07%
Epoch 10 | test loss 0.4205 | test acc 99.17%
Epoch 11 | test loss 0.3807 | test acc 99.20%
Epoch 12 | test loss 0.3534 | test acc 99.28%
Epoch 13 | test loss 0.3356 | test acc 99.31%
Epoch 14 | test loss 0.3237 | test acc 99.41%
Epoch 15 | test loss 0.3167 | test acc 99.46%
Epoch 16 | test loss 0.3119 | test acc 99.47%
Epoch 17 | test loss 0.3089 | test acc 99.50%
Epoch 18 | test loss 0.3069 | test acc 99.54%
Epoch 19 | test loss 0.3056 | test acc 99.54%
Epoch 20 | test loss 0.3047 | test acc 99.54%
Total parameters: 15634
```

## Inference

```
import torch
from mnist_model_v4 import CCMCCMCCP

model = CCMCCMCCP()
model.load_state_dict(torch.load("mnist_ccmccmccp_ema.pt"))
model.eval()

```
## Key Features

Compact (<20k params) but accurate (≥99.5%)

Global Average Pooling → avoids large FC layers

EMA + TTA for stable generalization

Flexible channel configuration
