Here’s a polished **README.md** that ties together your assignment breakdown and the logs from all three scenarios. I’ve structured it for clarity, with objectives, design, and results.

---

# CNN Architectures on MNIST

This project explores three progressively refined Convolutional Neural Network (CNN) architectures on the MNIST dataset. Each scenario emphasizes different aspects of model design: fundamentals, parameter efficiency, and robust performance.

---

## 📘 Assignment Breakdown

The work is divided into three scenarios:

1. **Scenario 1 – MNIST from Scratch**

   * Goal: Understand fundamentals of CNNs without pooling, batch normalization, or FC layers.
   * Architecture: Pure convolutional, only 3×3 kernels with same padding.
   * Learning: Receptive field calculation, intuition of feature aggregation.

2. **Scenario 2 – Parameter-Efficient CNN**

   * Goal: Reduce parameter count while maintaining >99% test accuracy.
   * Techniques: Introduce pooling, batch normalization, and 1×1 convolutions.
   * Learning: Trade-offs between accuracy and efficiency.

3. **Scenario 3 – High-Performance Consistent CNN**

   * Goal: Achieve ≥99.4% accuracy consistently within 15 epochs.
   * Techniques: Data augmentation, dropout, global average pooling (GAP), and one-cycle LR scheduling.
   * Learning: Regularization, augmentation, and stability.

---

## 🛠️ Experimental Setup

* **Dataset:** MNIST (28×28 grayscale digits, 10 classes)
* **Training:** 15 epochs per scenario
* **Optimizer:** SGD with variations (Scenario 3 used one-cycle policy)
* **Evaluation Metric:** Test accuracy

---

## 📊 Results

### Scenario 1 – From Scratch

* Simple CNN, no pooling, BN, or FC.
* Achieved **~99% accuracy by epoch 12**.
* Served as baseline for understanding receptive field growth.

Scenario 1 —  RF

Layers that affect RF (H×W after each):
Conv3×3 -> 26×26 -> r=3, j=1
Conv3×3 -> 24×24 -> r=5
Conv3×3 -> 22×22 -> r=7
MaxPool2×2,s2 -> 11×11 -> r=8, j=2
Conv1×1 -> 11×11 -> r=8
Conv3×3 -> 9×9 -> r=12
Conv3×3 -> 7×7 -> r=16
Conv1×1 -> 7×7 -> r=16
Conv7×7 -> 1×1 -> r=28

Final RF = 28 px (exactly the full 28×28 input).



---

### Scenario 2 – Parameter Efficient CNN

* Introduced max pooling, batch normalization, and 1×1 convolutions.
* Maintained **~99% accuracy** with **fewer parameters** compared to Scenario 1.
* Improved training stability and reduced computation.


Scenario 2 —  RF
```
Conv5×5,s1 -> *r = 1 + (5−1)1 = 5, j = 1
Conv3×3,s1  -> *r = 5 + (3−1)1 = 7, j = 1
MaxPool2×2,s2  -> *r = 7 + (2−1)1 = 8, j = 2
Conv3×3,s1  -> *r = 8 + (3−1)2 = 12, j = 2
Conv3×3,s1  -> *r = 12 + (3−1)2 = 16, j = 2
MaxPool2×2,s2  -> *r = 16 + (2−1)2 = 18, j = 4
Conv3×3,s1  -> *r = 18 + (3−1)4 = 26, j = 4
Conv3×3,s1  -> *r = 26 + (3−1)4 = 34, j = 4
GAP 7×7, s=7  -> *r = 34 + (7−1)4 = 58, j = 28
Linear -> r unchanged = 58

```
---

### Scenario 3 – High-Performance Consistent CNN

* Architecture summary:

  * Stacked Conv2D layers with BN + ReLU + Dropout
  * MaxPooling and AvgPooling
  * Final GAP + 1×1 Conv for classification
  * Total parameters: **7,544**

* Performance highlights:

  * **Epoch 3:** 99.02%
  * **Epoch 5:** 99.20%
  * **Epoch 7:** 99.37%
  * **Epoch 12:** 99.41%
  * **Epoch 14:** 99.44%

* Reached **≥99.4% accuracy consistently across runs**.

Scenario 3 —  RF
```
Conv3×3 -> r=3, j=1
Conv3×3 -> r=5
Conv1×1 -> r=5
MaxPool2×2 -> r=6, j=2
Conv3×3 -> r=10
Conv3×3 -> r=14
Conv3×3 -> r=18
Conv3×3 -> r=22
AvgPool 6×6, s=6 -> r=22 + (6−1)×2 = 32, j=12
Conv1×1 -> r=32

Final RF = 32 px (fully covers 28×28).
```
---

### Output

[Scenario 3 Output](./Scenario_3/Output.png)  
---

## 🔑 Key Learnings

1. **Scenario 1:** Fundamentals of CNNs, receptive fields, feature extraction without shortcuts.
2. **Scenario 2:** Efficiency via pooling, BN, and 1×1 convs; fewer params with high accuracy.
3. **Scenario 3:** Importance of augmentation, dropout, and LR scheduling for robustness.

---

## 📌 Conclusion

* MNIST can achieve >99% accuracy with even simple CNNs.
* Parameter efficiency does not necessarily reduce performance.
* Robustness and consistency require augmentation, regularization, and learning rate strategies.

---

## 📌  Logs

[Scenario 1 Log](./Scenario_1/scenario_1_log)  
[Scenario 2 Log](./Scenario_2/scenario_2_log)  
[Scenario 3 Log](./Scenario_3/Scenario_3_log)