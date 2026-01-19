import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ============================================================
# Paths
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(THIS_DIR, "..", "reports")
DATA_DIR = os.path.join(THIS_DIR, "..", "..", "datasets")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Dataset (MNIST)
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

# ============================================================
# Activation Factory
# ============================================================
def get_activation(name):
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    raise ValueError("Unknown activation")

# ============================================================
# Model
# ============================================================
class MLP(nn.Module):
    def __init__(self, activation, depth=8, width=256):
        super().__init__()
        layers = []
        act = get_activation(activation)

        layers.append(nn.Linear(28*28, width))
        layers.append(act)

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act)

        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# ============================================================
# Training & Evaluation
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()

    return loss_sum / len(loader), correct / len(loader.dataset)

# ============================================================
# Dead Neuron Measurement
# ============================================================
def dead_neuron_ratio(model, loader):
    model.eval()
    dead_ratios = []

    with torch.no_grad():
        for layer in model.net:
            if isinstance(layer, nn.ReLU):
                zero, total = 0, 0
                for x, _ in loader:
                    x = x.view(x.size(0), -1).to(device)
                    out = layer(x)
                    zero += (out == 0).sum().item()
                    total += out.numel()
                dead_ratios.append(zero / total)

    return np.mean(dead_ratios) if dead_ratios else 0.0

# ============================================================
# Experiment Runner
# ============================================================
activations = ["sigmoid", "tanh", "relu", "leaky_relu"]
results = {}

for act in activations:
    print(f"\nRunning activation: {act}")
    set_seed(42)

    model = MLP(act).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        _, acc = evaluate(model, val_loader, criterion)
        train_losses.append(loss)
        val_accs.append(acc)

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={acc:.4f}")

    dead_ratio = dead_neuron_ratio(model, train_loader)
    results[act] = (train_losses, val_accs, dead_ratio)

# ============================================================
# Plot: Training Loss
# ============================================================
plt.figure()
for act, (losses, _, _) in results.items():
    plt.plot(losses, label=act)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Activation Comparison — Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "activation_training_loss.png"))
plt.close()

# ============================================================
# Plot: Dead Neurons
# ============================================================
labels = []
values = []

for act, (_, _, dead) in results.items():
    if act in ["relu", "leaky_relu"]:
        labels.append(act)
        values.append(dead)

plt.figure()
plt.bar(labels, values)
plt.ylabel("Dead Neuron Ratio")
plt.title("Dead Neurons: ReLU vs Leaky ReLU")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "dead_neurons.png"))
plt.close()

print("\nSaved plots to:", REPORT_DIR)
