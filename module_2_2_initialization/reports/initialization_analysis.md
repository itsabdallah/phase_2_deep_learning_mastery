# Assignment 2.2.A — Weight Initialization & Signal Propagation

## Objective

The goal of this experiment is to empirically study how **weight initialization affects signal propagation** in deep neural networks. Specifically, we analyze how different initialization schemes influence:

* Activation variance across depth
* Gradient norm behavior across depth

This helps explain *why deep networks suffer from vanishing or exploding gradients* and how principled initialization methods mitigate these issues.

---

## Experimental Setup

* **Model:** Deep MLP with 30 hidden layers
* **Width:** 256 neurons per layer
* **Activations:** ReLU or Tanh (matched to initialization)
* **Training:** No training performed (single forward + backward pass)
* **Input:** Random Gaussian input batch

### Initialization Schemes Compared

1. **Bad Initialization (std = 0.01)**
2. **Bad Initialization (std = 1.0)**
3. **Xavier Initialization (Tanh)**
4. **He Initialization (ReLU)**

---

## Results & Observations

![Training Loss](activation_statistics.png)


![Training Loss](gradient_statistics.png)

### 1. Bad Initialization (std = 0.01)

* **Activation Variance:** Rapidly collapses toward zero as depth increases.
* **Gradient Norms:** Gradients vanish almost completely in deeper layers.
* **Interpretation:** Very small initial weights shrink the signal at every layer. With depth, this multiplicative effect causes both activations and gradients to disappear, making learning impossible.

---

### 2. Bad Initialization (std = 1.0)

* **Activation Variance:** Explodes quickly with depth.
* **Gradient Norms:** Extremely large gradients, leading to instability.
* **Interpretation:** Large initial weights amplify the signal at each layer. This causes exponential growth of activations and gradients, resulting in numerical instability and unreliable optimization.

---

### 3. Xavier Initialization (Tanh)

* **Activation Variance:** Remains relatively stable across depth.
* **Gradient Norms:** Gradients are preserved without vanishing or exploding.
* **Interpretation:** Xavier initialization scales weights based on fan-in and fan-out, maintaining variance through layers. This is well-matched to symmetric activations like Tanh, enabling stable signal propagation in deep networks.

---

### 4. He Initialization (ReLU)

* **Activation Variance:** Stable across layers despite network depth.
* **Gradient Norms:** Large but controlled gradients that do not vanish.
* **Interpretation:** He initialization accounts for the fact that ReLU deactivates half of the neurons on average. By increasing variance accordingly, it preserves signal magnitude and enables effective gradient flow even in very deep networks.

---

## Key Takeaways

* Depth amplifies poor initialization choices, leading to vanishing or exploding gradients.
* Proper initialization is **a prerequisite for training deep networks**, independent of optimizer choice.
* Xavier initialization is suited for Tanh-like activations.
* He initialization is essential for ReLU-based deep networks.

> **Conclusion:** Even with perfect optimizers and learning rates, deep networks cannot train successfully unless signal propagation is stabilized through proper initialization.

---

## Artifacts

* `activation_statistics.png`
* `gradient_statistics.png`

These plots visually confirm the theoretical behavior of different initialization strategies in deep neural networks.
