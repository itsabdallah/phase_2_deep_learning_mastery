# Assignment 2.4.A — Weight Decay (L2 Regularization)

## Objective

This experiment studies the effect of **weight decay (L2 regularization)** on training dynamics, generalization, and parameter magnitudes. We compare three settings:

* **WD = 0.0** (no regularization)
* **WD = 1e-4** (mild regularization)
* **WD = 1e-3** (stronger regularization)

All other factors (architecture, optimizer, learning rate, initialization, dataset, and number of epochs) are kept constant to ensure a fair comparison.

---

## Experimental Setup

* **Model**: Simple feedforward neural network (same as previous Phase 2 experiments)
* **Optimizer**: Adam
* **Loss Function**: Cross-Entropy Loss
* **Epochs**: 10
* **Metrics Tracked**:

  * Training loss
  * Validation loss
  * Validation accuracy
  * Final weight norms

Weight decay is applied via the optimizer as standard L2 regularization.

---

## Results & Observations

### 1. Training Loss Behavior

From `weight_decay_training_loss.png`:

* All three configurations converge smoothly.
* **WD = 0.0** achieves the *lowest training loss*, as expected, since there is no penalty on large weights.
* Increasing weight decay slightly **slows down training convergence**, with **WD = 1e-3** showing the highest training loss.

**Interpretation:**
Weight decay explicitly penalizes large weights, preventing the model from over-optimizing the training set. This manifests as slightly higher training loss for larger WD values.

---

### 2. Validation Loss Behavior

From `weight_decay_validation_loss.png`:

* Validation loss decreases for all settings.
* **WD = 1e-4** achieves the *lowest final validation loss*.
* **WD = 0.0** shows mild instability in later epochs (slight increase), indicating early signs of overfitting.
* **WD = 1e-3** remains stable but converges slightly slower.

**Interpretation:**
Moderate regularization improves generalization by discouraging overly complex solutions. Too little regularization risks overfitting, while too much can underfit.

---

### 3. Validation Accuracy

From `weight_decay_validation_accuracy.png`:

* All models reach high accuracy (>97%).
* **WD = 1e-4** achieves the *best peak validation accuracy*.
* **WD = 1e-3** catches up by the end but shows slower early progress.
* **WD = 0.0** plateaus slightly earlier.

**Interpretation:**
Weight decay improves generalization without sacrificing accuracy when tuned correctly. The differences are subtle but consistent.

---

### 4. Effect on Weight Norms

From `weight_norms.png`:

* **WD = 0.0** produces the *largest weight norms*.
* **WD = 1e-4** slightly reduces weight magnitudes.
* **WD = 1e-3** significantly constrains weight growth.

**Interpretation:**
This confirms the theoretical role of L2 regularization: it biases the optimization toward smaller-norm solutions, which are often associated with better generalization.

---

## Key Takeaways

1. **Weight decay does not primarily improve training loss** — it improves *generalization*.
2. **Moderate weight decay (1e-4)** provides the best trade-off between bias and variance in this setup.
3. Excessive regularization (**1e-3**) constrains the model too strongly early in training.
4. Weight norms provide a concrete, measurable explanation for the observed behavior.

---

## Practical Guidelines

* Always tune weight decay jointly with learning rate.
* Start with **1e-4** as a strong baseline for Adam-based optimizers.
* Monitor both **validation loss** and **weight norms**, not just accuracy.
* Weight decay complements (but does not replace) normalization and proper initialization.

---

## Conclusion

This experiment demonstrates that weight decay acts as an effective capacity control mechanism. While it slightly increases training loss, it leads to smoother optimization, smaller weight magnitudes, and improved validation performance. These results align with both classical statistical learning theory and modern deep learning practice.

This completes **Assignment 2.4.A** and prepares the ground for more advanced regularization techniques in subsequent modules.
