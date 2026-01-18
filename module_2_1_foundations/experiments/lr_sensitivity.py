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
# Paths (robust, absolute, reproducible)
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(THIS_DIR, "..", "reports")
DATA_DIR = os.path.join(THIS_DIR, "..", "datasets")

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
print("Using device:", device)


# ============================================================
# Dataset (MNIST)
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)


# ============================================================
# Model Definition
# ============================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

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
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


# ============================================================
# Experiment Runner
# ============================================================
def run_experiment(name, optimizer_fn, lr=1e-3, epochs=10):
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"[{name}] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return train_losses, val_losses, val_accuracies

# ============================================================
# Learning Rate Sensitivity Experiment
# ============================================================
learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
results = {}

for lr in learning_rates:
    set_seed(42)

    name = f"SGD+Momentum (lr={lr})"
    optimizer_fn = lambda params, lr=lr: optim.SGD(
        params, lr=lr, momentum=0.9
    )

    results[lr] = run_experiment(
        name=name,
        optimizer_fn=optimizer_fn,
        lr=lr,
        epochs=10
    )



# ============================================================
# Plot: Training Loss
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for lr, (train_losses, _, _) in results.items():
    ax.plot(train_losses, label=f"lr={lr}")

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Learning Rate Sensitivity — SGD+Momentum")
ax.legend()
ax.grid(True)

fig.tight_layout()
save_path = os.path.join(REPORT_DIR, "lr_sensitivity_training_loss.png")
fig.savefig(save_path)
plt.close(fig)

print("Saved training loss plot to:", os.path.abspath(save_path))
