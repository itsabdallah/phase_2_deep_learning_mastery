# Analysis: Activation Functions (Assignment 2.2.B)

## 1. Training Dynamics Across Activations

From the training loss curves, clear differences emerge between activation functions:

- **Sigmoid** shows almost no learning progress. The training loss remains high and flat across epochs, indicating **severe vanishing gradient issues** in deeper networks.

- **Tanh** learns faster than sigmoid but still converges more slowly than ReLU-based activations. Its symmetric output helps optimization, but saturation at large magnitudes still limits gradient flow.

- **ReLU** converges rapidly with low training loss but introduces instability due to inactive neurons.

- **Leaky ReLU** demonstrates **fast convergence and stable training**, achieving low loss consistently across epochs.

This confirms that activation choice has a direct impact on optimization speed and stability, especially in deeper architectures.

## 2. Dead Neuron Analysis (ReLU vs Leaky ReLU)

The dead neuron comparison highlights a critical limitation of standard ReLU:

- **ReLU** exhibits a very high dead neuron ratio (≈80%).

    - Many neurons output zero for all inputs due to negative pre-activations.

    - Once dead, these neurons stop learning entirely because gradients are zero.

- **Leaky ReLU** shows **near-zero dead neurons**.

    - The small negative slope ensures gradients continue to flow even when activations are negative.

This explains why Leaky ReLU maintains better representational capacity over training and avoids capacity collapse.

## 3. Validation Performance (Leaky ReLU)

Leaky ReLU achieves strong and stable validation accuracy:

- Validation accuracy quickly rises above **94%** in the first epoch.

- Peaks around **97.1%**, remaining stable throughout training.

- Minor fluctuations at later epochs suggest healthy regularization rather than overfitting.

This indicates that Leaky ReLU not only optimizes well but also **generalizes effectively**.

## 4. Theoretical Alignment

The experimental results align closely with theory:

- **Sigmoid / Tanh**
    → Suffer from vanishing gradients due to saturation.

- **ReLU**
    → Enables fast training but risks dead neurons.

- **Leaky ReLU**
    → Preserves gradient flow and avoids neuron death.

Thus, Leaky ReLU provides a **balanced trade-off** between optimization efficiency and network capacity.

## 5. Key Takeaways

- Activation functions are a **first-order design choice**, not a minor detail.

- Poor activation selection can completely stall learning (sigmoid).

- ReLU improves optimization but introduces structural risks.

- **Leaky ReLU offers the most robust behavior**, combining:

    - Fast convergence

    - Stable gradients

    - Minimal dead neurons

    - Strong generalization

## 6. Conclusion

This assignment demonstrates that activation functions play a **critical role in deep network training,** affecting optimization speed, gradient flow, and effective model capacity.

The experiments show that **sigmoid activations are unsuitable for deep networks,** as training loss remained high and stagnant due to severe vanishing gradient effects. Tanh improves over sigmoid by centering activations but still suffers from saturation, resulting in slower convergence.

ReLU significantly accelerates training and achieves low loss; however, it introduces a major structural weakness: ****dead neurons**. A large proportion of ReLU neurons became inactive early in training, permanently reducing the network’s expressive power.

Leaky ReLU consistently outperformed all other activations in terms of **stability, convergence, and robustness.** It preserved gradient flow for negative inputs, nearly eliminated dead neurons, and achieved strong validation accuracy with smooth training dynamics.

Overall, the results strongly align with theoretical expectations and confirm that **Leaky ReLU provides the best balance between optimization efficiency and representational stability.** This makes it the most reliable default activation for deep fully connected networks.


## Final Takeaways

- Activation choice directly impacts gradient propagation and learning stability.

- Saturating activations (sigmoid, tanh) hinder deep learning.

- ReLU accelerates training but risks capacity collapse.

- **Leaky ReLU is the most robust activation tested,** combining fast convergence, stable gradients, and strong generalization.




## Artifacts

* `activation_training_loss.png`
* `dead_neurons.png`