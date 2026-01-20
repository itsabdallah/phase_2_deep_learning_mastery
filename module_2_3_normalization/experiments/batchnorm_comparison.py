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
# Model Definitions
# ============================================================
class MLP_NoBN(nn.Module):
    """Deep MLP without normalization"""
    def __init__(self, depth=6, width=256):
        super().__init__()
        layers = []
        in_dim = 28 * 28

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width

        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class MLP_BN(nn.Module):
    """Deep MLP with Batch Normalization"""
    def __init__(self, depth=6, width=256):
        super().__init__()
        layers = []
        in_dim = 28 * 28

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            in_dim = width

        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

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

    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


# ============================================================
# Experiment Runner
# ============================================================
def run_experiment(name, model, lr=0.1, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)

        print(
            f"[{name}] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return train_losses


# ============================================================
# Run Comparison
# ============================================================
set_seed(42)
no_bn_model = MLP_NoBN().to(device)
bn_model = MLP_BN().to(device)

results = {}
results["No BatchNorm"] = run_experiment(
    "No BatchNorm", no_bn_model, lr=0.1, epochs=10
)

set_seed(42)
results["BatchNorm"] = run_experiment(
    "BatchNorm", bn_model, lr=0.1, epochs=10
)


# ============================================================
# Plot Results
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for name, losses in results.items():
    ax.plot(losses, label=name)

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("BatchNorm vs No Normalization — Training Loss")
ax.legend()
ax.grid(True)

fig.tight_layout()
save_path = os.path.join(REPORT_DIR, "batchnorm_training_loss.png")
fig.savefig(save_path)
plt.close(fig)

print("Saved plot to:", os.path.abspath(save_path))
