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
