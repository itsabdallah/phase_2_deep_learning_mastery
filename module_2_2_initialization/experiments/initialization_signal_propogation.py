import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(THIS_DIR, "..", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Deep MLP Definition
# ============================================================
class DeepMLP(nn.Module):
    def __init__(self, depth=30, width=256, activation="relu"):
        super().__init__()

        layers = []
        act = nn.ReLU() if activation == "relu" else nn.Tanh()

        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(act)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        activations = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, (nn.ReLU, nn.Tanh)):
                activations.append(x)
        return x, activations

# ============================================================
# Initialization Methods
# ============================================================
def bad_init(model, std):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            nn.init.zeros_(m.bias)

def xavier_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

def he_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

# ============================================================
# Statistics Collection
# ============================================================
def activation_stats(activations):
    means = [a.mean().item() for a in activations]
    variances = [a.var().item() for a in activations]
    return means, variances

def gradient_norms(model):
    norms = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            norms.append(m.weight.grad.norm().item())
    return norms

# ============================================================
# Experiment Runner
# ============================================================
def run_experiment(name, init_fn, activation):
    set_seed(42)

    model = DeepMLP(depth=30, width=256, activation=activation).to(device)
    init_fn(model)

    x = torch.randn(128, 256).to(device)
    out, activations = model(x)

    loss = out.sum()
    loss.backward()

    act_means, act_vars = activation_stats(activations)
    grad_norms = gradient_norms(model)

    return act_vars, grad_norms

# ============================================================
# Run Experiments
# ============================================================
experiments = {
    "Bad Init (std=0.01)": lambda m: bad_init(m, 0.01),
    "Bad Init (std=1.0)": lambda m: bad_init(m, 1.0),
    "Xavier (tanh)": xavier_init,
    "He (ReLU)": he_init,
}

results = {}

for name, init_fn in experiments.items():
    activation = "tanh" if "Xavier" in name else "relu"
    print(f"Running: {name}")
    results[name] = run_experiment(name, init_fn, activation)

# ============================================================
# Plot: Activation Variance vs Depth
# ============================================================
plt.figure(figsize=(8, 6))

for name, (vars_, _) in results.items():
    plt.plot(vars_, label=name)

plt.xlabel("Layer Depth")
plt.ylabel("Activation Variance")
plt.title("Activation Variance vs Depth")
plt.legend()
plt.grid(True)

act_path = os.path.join(REPORT_DIR, "activation_statistics.png")
plt.savefig(act_path)
plt.close()

# ============================================================
# Plot: Gradient Norm vs Depth
# ============================================================
plt.figure(figsize=(8, 6))

for name, (_, grads) in results.items():
    plt.plot(grads, label=name)

plt.xlabel("Layer Depth")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm vs Depth")
plt.legend()
plt.grid(True)

grad_path = os.path.join(REPORT_DIR, "gradient_statistics.png")
plt.savefig(grad_path)
plt.close()

print("Saved plots:")
print(act_path)
print(grad_path)
