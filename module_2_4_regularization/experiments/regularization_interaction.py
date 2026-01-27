# ============================================================
# Regularization Interaction Experiment
# L2 (Weight Decay) vs Dropout vs Both
# ============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# CRITICAL: Force non-interactive backend (Windows-safe)
# ------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.01
WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
val_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_P) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ------------------------------------------------------------
# Training & Evaluation
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
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
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)

            total_loss += criterion(outputs, y).item()
            correct += (outputs.argmax(dim=1) == y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

# ------------------------------------------------------------
# Experiments
# ------------------------------------------------------------
experiments = {
    "No Regularization": {
        "dropout": False,
        "weight_decay": 0.0
    },
    "Weight Decay Only": {
        "dropout": False,
        "weight_decay": WEIGHT_DECAY
    },
    "Dropout Only": {
        "dropout": True,
        "weight_decay": 0.0
    },
    "Dropout + Weight Decay": {
        "dropout": True,
        "weight_decay": WEIGHT_DECAY
    }
}

results = {}

criterion = nn.CrossEntropyLoss()

for name, cfg in experiments.items():
    print(f"\nRunning experiment: {name}")

    model = MLP(use_dropout=cfg["dropout"]).to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        weight_decay=cfg["weight_decay"]
    )

    results[name] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        results[name]["train_loss"].append(train_loss)
        results[name]["val_loss"].append(val_loss)
        results[name]["val_acc"].append(val_acc)

        print(
            f"[{name}] Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

# ------------------------------------------------------------
# Plotting (SAFE & ROBUST)
# ------------------------------------------------------------
epochs = range(1, EPOCHS + 1)

def plot_metric(key, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data in results.items():
        ax.plot(epochs, data[key], label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.canvas.draw()  # force render

    save_path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {save_path}")

plot_metric(
    "train_loss",
    "Training Loss",
    "regularization_interaction_training_loss.png"
)

plot_metric(
    "val_loss",
    "Validation Loss",
    "regularization_interaction_validation_loss.png"
)

plot_metric(
    "val_acc",
    "Validation Accuracy",
    "regularization_interaction_validation_accuracy.png"
)

print("\nAll experiments completed successfully.")
