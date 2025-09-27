
---

# Scenario 1 — **MNIST from Scratch**

### Objective - (Target)

* Achieve **>99% test accuracy within 15 epochs** using a very simple CNN built from scratch.
* Primary goal: **understand fundamentals** of convolutional networks and image representations.

### Constraints and Assumptions

* Dataset: MNIST (grayscale, single channel).
* Input dimensions: explicitly inspect and confirm (height × width × channels).
* No color channels (1 channel only).
* Kernels fixed at **3×3** with **same padding**.
* Use **only CNN layers**, no fully connected layers.
* Transition to logits done via FC or **1×1 convolution** at the end.
* Exclude advanced techniques:

  * No max pooling
  * No strided convolutions
  * No batch normalization
  * No GAP (Global Average Pooling)

### Design Choices

* Use **basic normalization** for inputs.
* Entire architecture is convolutional.
* Receptive field must be calculated layer by layer (important for understanding how local features aggregate).

### Learning Outcomes

* Grasp **image dimensionality** and channel concepts.
* Learn how CNNs without pooling/FC layers can still classify.
* Build intuition on **receptive field growth** and how stacking 3×3 kernels builds global understanding.
