
# Scenario 2 — **Parameter-Efficient CNN**

### Objective - (Target)

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
