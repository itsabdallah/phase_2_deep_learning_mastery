# Assignment 2.4.C – Regularization Interaction

## Objective

The goal of this experiment is to study how **different regularization techniques interact when applied together**. While individual regularization methods such as **weight decay** and **dropout** are well understood in isolation, their combined effect is not always additive. This assignment empirically investigates whether combining these methods improves generalization or leads to diminishing returns.

Specifically, we compare four training setups:

1. No regularization
2. Weight decay only
3. Dropout only
4. Dropout + weight decay

---

## Experimental Setup

### Model Architecture

* Fully connected neural network (MLP)
* ReLU activations
* Dropout applied after hidden layers (when enabled)

### Training Configuration

* Optimizer: Adam
* Learning rate: Fixed across all experiments
* Epochs: 10
* Dataset: Same train/validation split for all runs
* Device: CPU

### Regularization Configurations

| Experiment             | Weight Decay | Dropout |
| ---------------------- | ------------ | ------- |
| No Regularization      | ❌            | ❌       |
| Weight Decay Only      | ✅            | ❌       |
| Dropout Only           | ❌            | ✅       |
| Dropout + Weight Decay | ✅            | ✅       |

All other hyperparameters were held constant to ensure a fair comparison.

---

## Metrics Tracked

For each experiment, the following metrics were recorded at every epoch:

* Training loss
* Validation loss
* Validation accuracy

Corresponding plots were generated and saved for analysis.

---

## Results & Analysis

### Training Loss

**Observation:**

* The model without regularization achieved the lowest training loss.
* Dropout (with or without weight decay) resulted in higher training loss throughout training.

**Interpretation:**
Regularization techniques intentionally restrict the model’s ability to fit the training data. Dropout introduces stochastic noise by randomly disabling neurons, while weight decay penalizes large weights. Both mechanisms slow down optimization, which explains the higher training loss.

This behavior is expected and confirms correct implementation.

---

### Validation Loss

**Observation:**

* Dropout-only and Dropout + Weight Decay achieved the lowest validation loss.
* Weight decay alone provided modest improvement compared to no regularization.
* Combining dropout and weight decay did not yield a dramatic improvement over dropout alone.

**Interpretation:**
Dropout is particularly effective at preventing feature co-adaptation, leading to stronger generalization. Weight decay smooths the parameter space but does not directly address co-adaptation. When combined, the two techniques partially overlap in effect, resulting in diminishing returns rather than additive gains.

---

### Validation Accuracy

**Final Accuracy Comparison:**

* No Regularization: ~94.9%
* Weight Decay Only: ~94.8%
* Dropout Only: ~95.15%
* Dropout + Weight Decay: ~95.18%

**Interpretation:**
Dropout produced the largest standalone improvement in validation accuracy. Weight decay alone offered limited gains, while the combined setup achieved only a marginal improvement over dropout.

This highlights that stronger regularization does not always imply better performance.

---

## Key Insight: Regularization Is Not Additive

A critical takeaway from this experiment is that **regularization methods interact rather than stack linearly**.

* Weight decay controls parameter magnitude.
* Dropout reduces reliance on specific neurons.
* Applying both simultaneously can lead to overlapping constraints.

Beyond a certain point, adding more regularization increases optimization difficulty without meaningful gains in generalization.

---

## Conclusion

This experiment demonstrates that while both weight decay and dropout improve generalization individually, their combined effect yields diminishing returns. Dropout emerged as the most effective single regularization technique in this setting, while weight decay provided secondary benefits.

The results emphasize the importance of understanding *how* regularization techniques influence learning dynamics, rather than blindly combining them. Effective regularization requires balance, empirical validation, and awareness of interaction effects.

---

## Takeaways for Practice

* Always evaluate regularization techniques both individually and jointly
* Monitor training loss to detect over-regularization
* Do not assume that combining techniques guarantees better performance

This assignment reinforces the principle that **regularization is a design choice, not a checklist item**.
