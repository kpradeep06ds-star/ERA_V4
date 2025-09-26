

---

# Scenario 1 — **MNIST from Scratch**

### Objective

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

---

# Scenario 2 — **Parameter-Efficient CNN**

### Objective

* Maintain **>99% test accuracy within 15 epochs** but with **significantly fewer parameters**.
* Optimize for **efficiency and compactness**.

### Constraints and Assumptions

* Begin with architecture from **Scenario 1**.
* Explicit focus on **reducing parameter count**.

### Design Choices

* Introduce **max pooling** to reduce spatial dimensions and hence parameters.
* Use **batch normalization** to stabilize and speed up convergence.
* Replace large convolutions with **1×1 convolutions** for dimensionality reduction where useful.
* Output stage:  use either **1×1 convolution** or **GAP** or **FC** or mix for final classification.

### Learning Outcomes

* Understand **trade-off between accuracy and parameter count**.
* Learn role of:

  * **Pooling** (spatial reduction, invariance).
  * **Batch normalization** (stabilized learning).
  * **1×1 convolutions** (parameter bottlenecking, dimension adjustment).
* Compare receptive field calculations with Scenario 1 and notice efficiency gains.

---

# Scenario 3 — **High-Performance Consistent CNN**

### Objective

* Push accuracy to **>=99.4% consistently across multiple runs/epochs** within 15 epochs.
* Focus on **generalization, regularization, and training stability**.

### Constraints and Assumptions

* Build on compact architecture of **Scenario 2**.
* Emphasis on **robustness across epochs**, not just peak accuracy.

### Design Choices

* Data augmentation:

  * Random rotations
  * Small translations / random cropping
  * Other mild transformations
* Add **regularization**:

  * Dropout (p=0.05–0.10) for robustness.
* Final stage:

  * Use **Global Average Pooling (GAP)** instead of fully connected layers.
  * Optimization with **SGD** + **one-cycle learning rate schedule** for smoother training.
* Continue with **3×3 kernels** (default) unless absolutely necessary to change.
* Receptive field should again be tracked and compared with earlier scenarios.

### Learning Outcomes

* Explore **data augmentation** as a way to expand dataset information.
* Understand **dropout’s role** in combating overfitting.
* See how **learning rate scheduling** improves convergence and consistency.
* Learn to balance **complexity vs regularization** for stable high accuracy.

---

# Summary Across Scenarios

* **Scenario 1:** Build the foundation — simple CNN, no tricks, understand dimensions and receptive field.
* **Scenario 2:** Optimize for efficiency — fewer parameters, pooling, batchnorm, 1×1 conv.
* **Scenario 3:** Optimize for robustness and consistency — augment data, add dropout, use GAP + SGD + learning rate scheduling.

Each scenario is a **module** of increasing sophistication, ensuring that by the end you’ve mastered:

1. Basics of CNN structure and receptive field (Scenario 1).
2. Efficiency and architectural trade-offs (Scenario 2).
3. Generalization and stability techniques (Scenario 3).

---


