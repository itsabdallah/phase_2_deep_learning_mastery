# Dropout vs Weight Decay — Regularization Comparison

## Assignment 2.4.B

### Objective

The goal of this experiment is to **compare Dropout and Weight Decay (L2 regularization)** as regularization techniques in deep neural networks. Both methods aim to reduce overfitting, but they operate through fundamentally different mechanisms. This assignment evaluates their effects on:

* Training loss
* Validation loss
* Validation accuracy
* Convergence behavior

The experiment is conducted under controlled conditions using the same architecture, dataset, optimizer, learning rate, and number of epochs.

---

## Conceptual Background

### Dropout

Dropout is a **stochastic regularization technique**. During training, it randomly disables (sets to zero) a fraction of neuron activations.

**Key properties:**

* Forces redundancy in representations
* Prevents co-adaptation of neurons
* Acts like training an ensemble of subnetworks
* Only active during training (disabled at inference)

**Expected behavior:**

* Higher training loss
* Strong regularization effect
* Improved generalization when overfitting is present

---

### Weight Decay (L2 Regularization)

Weight decay adds a penalty proportional to the squared magnitude of weights to the loss function.

**Key properties:**

* Encourages smaller weights
* Smooths the loss landscape
* Acts continuously throughout training
* Integrated directly into the optimizer

**Expected behavior:**

* Lower training loss than dropout
* More stable and faster convergence
* Strong bias toward simpler models

---

## Experimental Setup

* **Dataset:** MNIST
* **Model:** Fully-connected neural network
* **Optimizer:** Adam
* **Learning Rate:** Fixed
* **Epochs:** 10
* **Batch Size:** Fixed

Two models were trained:

1. **Dropout model** — includes dropout layers
2. **Weight decay model** — uses L2 regularization via optimizer

Only one regularization method is active at a time.

---

## Results

### Training Loss

From `dropout_vs_weight_decay_training_loss.png`:

The training loss curve shows:

* Dropout maintains **higher training loss** across all epochs
* Weight decay converges faster and reaches lower training loss

**Interpretation:**
Dropout injects noise into training by randomly disabling neurons, making optimization harder. Weight decay, in contrast, preserves full network capacity while softly constraining parameters.

---

### Validation Loss

From `dropout_vs_weight_decay_validation_loss.png`:

Both models show a steady decrease in validation loss:

* Weight decay achieves slightly lower validation loss early on
* Dropout closely tracks weight decay in later epochs

**Interpretation:**
Both regularizers successfully prevent overfitting. Weight decay provides smoother early generalization, while dropout catches up as training progresses.

---

### Validation Accuracy

From `dropout_vs_weight_decay_validation_accuracy.png`:

Validation accuracy improves consistently for both approaches:

* Final accuracy is **nearly identical** (~95%)
* Weight decay slightly outperforms dropout in early epochs
* Dropout shows competitive generalization despite higher training loss

**Interpretation:**
Although dropout harms training performance, it does not reduce final generalization. Both methods are effective at preventing overfitting.

---

## Key Observations

1. **Training difficulty**

   * Dropout increases optimization difficulty
   * Weight decay preserves smoother gradients

2. **Generalization**

   * Both techniques generalize equally well in this setting

3. **Convergence speed**

   * Weight decay converges faster
   * Dropout slows early training

4. **Regularization strength**

   * Dropout acts strongly and stochastically
   * Weight decay acts gently and continuously

---

## Practical Takeaways

* Use **weight decay** when:

  * Training stability and fast convergence are important
  * Model is moderately over-parameterized

* Use **dropout** when:

  * Severe overfitting is observed
  * Redundancy and robustness are desired

* In modern deep learning practice:

  * Weight decay is often the default
  * Dropout is used selectively or sparingly

---

## Conclusion

This experiment demonstrates that **Dropout and Weight Decay achieve similar generalization performance through different mechanisms**. Weight decay provides smoother optimization and faster convergence, while dropout enforces robustness at the cost of higher training loss.

Understanding these trade-offs is critical when designing regularization strategies for deep neural networks, especially in larger and more complex models.

---

**Status:** Assignment 2.4.B — Completed
