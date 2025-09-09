# MNIST-CNN (Tiny) — README

A compact, easy-to-train convolutional neural network for MNIST, with just \~24k parameters and light augmentation. The goal is to keep the model small and readable while retaining solid accuracy on 28×28 grayscale digits.&#x20;

---

## Overview

* **Frameworks:** PyTorch + TorchVision
* **File:** `mnist_model.py` (single-file training script)
* **Dataset:** MNIST (auto-downloaded)
* **Params:** \~24,012
* **Default run:** 1 epoch, batch size 64, Adam (lr=0.01)

---

## Architecture

`MnistCNN` (all shapes assume input `N×1×28×28`)

| Stage | Layer                                                 |            Output shape | Notes                                     |
| ----- | ----------------------------------------------------- | ----------------------: | ----------------------------------------- |
| 1     | `Conv2d(1→14, k3, p1)` + **MaxPool2d(2)** + **GELU**  |            `N×14×14×14` | 3×3 keeps spatial size, pool halves 28→14 |
| 2     | `Conv2d(14→26, k3, p1)` + **MaxPool2d(2)** + **GELU** |              `N×26×7×7` | second conv stack                         |
| 3     | `Flatten`                                             | `N×(26·7·7)` = `N×1274` | reshape for FC                            |
| 4     | `Linear(1274→16)` + **ReLU**                          |                  `N×16` | small dense bottleneck                    |
| 5     | `Linear(16→10)`                                       |                  `N×10` | class logits                              |
| 6     | `log_softmax(dim=1)`                                  |                  `N×10` | numerically stable log-probs              |

> **Parameter breakdown (approx.)**
> Conv1: 140 • Conv2: 3,302 • FC1: 20,400 • FC2: 170 → **Total ≈ 24,012**

**Forward path:**
`x → conv1 → maxpool → GELU → conv2 → maxpool → GELU → flatten → FC1 → ReLU → FC2 → log_softmax`

---

## Data pipeline

Defined in `get_data_transforms()`:

* **Train transforms**

  * `RandomRotation(±2°)`
  * `RandomAffine(translate=±1%)`
  * `RandomAffine(shear=±2°)` *(applied as a separate call)*
  * `ToTensor()`
  * `Normalize(mean=0.5, std=0.5)` → scales to \~\[-1, 1]

* **Test transforms**

  * `ToTensor()`, `Normalize(0.5, 0.5)`

The augmentations are intentionally **tiny** to preserve digit identity while improving robustness.

---

## Training loop

* **Optimizer:** `Adam(lr=0.01)`
* **Loss:** `CrossEntropyLoss`
* **Metrics:** prints running loss; per-epoch train/test loss & accuracy
* **Return values:**

  * `train_model(epochs=E, batch_size=B, return_both_accuracies=False)`
  * Returns `(model, test_acc)` by default, or `(model, train_acc, test_acc)` if requested.

> **Note on criterion vs. model output**
> The model returns **log-softmax** outputs, while the training uses `CrossEntropyLoss` (which expects raw logits and applies log-softmax internally). Functionally it still learns, but for strict correctness you can either:
>
> * change the final layer to return raw logits (remove `log_softmax`) **or**
> * keep `log_softmax` and switch the loss to `NLLLoss`.

---

## Quickstart

### 1) Install deps

```bash
pip install -r requirements.txt
```

### 2) Train (default: 1 epoch)

```bash
python mnist_model.py
```

You’ll see training/test metrics and a final parameter count.

### 3) Use as a module

```python
from mnist_model import train_model
model, test_acc = train_model(epochs=3, batch_size=128)
```

---

## Design choices (why this works)

* **Small channel counts (14/26) & tiny FC (16 units):** keeps params and overfitting in check, great for CPU training.
* **Two conv+pool stages:** shrink 28→14→7 while adding nonlinearity and local feature extraction.
* **GELU in conv blocks + ReLU in head:** smooth activations early, crisp decision head later.
* **Light augmentation:** rotation/translation/shear nudges generalization without distorting digits.

---

## Potential improvements

* Swap `CrossEntropyLoss` ↔ `NLLLoss` (or remove `log_softmax`) for criterion–output consistency.
* Re-enable dropout (commented) if you see overfitting.
* Add **torch.backends.cudnn.benchmark** and GPU `.to(device)` plumbing for speed.
* Track metrics with `torchmetrics` or `tensorboard`.
* Set random seeds for reproducibility and log them.

---

## File map

* `mnist_model.py` — model, transforms, train/eval entrypoint (single-file solution).&#x20;

---

## License
MIT

## Logs

```
 WIN11@DESKTOP-QUJGUHV  D:  ERV_V4  4  master   env10 3.10.9 
❯ python .\mnist_model.py
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.325199
❯ python .\mnist_model.py
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.325199
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.301803
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.301803
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.099910
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.099910
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.006863
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.006863
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.040822
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.087675
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.179319
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.134739
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.078868
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.162999

Training set: Average loss: 0.1426, Accuracy: 57345/60000 (95.58%)

Test set: Average loss: 0.0678, Accuracy: 9785/10000 (97.85%)


Total number of parameters: 24012
❯ python .\mnist_model.py
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.305994
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.202221
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.087447
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.081993
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.034541
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.059910
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.060663
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.033077
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.209164
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.094817

Training set: Average loss: 0.1528, Accuracy: 57133/60000 (95.22%)

Test set: Average loss: 0.0784, Accuracy: 9787/10000 (97.87%)


Total number of parameters: 24012

❯ python .\mnist_model.py
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.340797
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.185377
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.257567
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.167317
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.252204
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.065720
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.172130
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.026657
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.089293
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.190713

Training set: Average loss: 0.1526, Accuracy: 57246/60000 (95.41%)

Test set: Average loss: 0.0654, Accuracy: 9801/10000 (98.01%)


Total number of parameters: 24012
```