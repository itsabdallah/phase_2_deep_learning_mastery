import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------
# Hyperparameters
# ---------------------------
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3]

# ---------------------------
# Dataset
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_set, val_set = random_split(full_train, [50000, 10000])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)

    return avg_loss, acc


def compute_weight_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        total_norm += p.data.norm(2).item() ** 2
    return total_norm ** 0.5

# ---------------------------
# Experiment Loop
# ---------------------------
criterion = nn.CrossEntropyLoss()

results = {}

for wd in WEIGHT_DECAYS:
    print(f"\nRunning experiment with weight_decay={wd}")

    model = MLP().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=wd
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"[WD={wd}] Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    weight_norm = compute_weight_norm(model)

    results[wd] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "weight_norm": weight_norm
    }

# ---------------------------
# Plotting
# ---------------------------
epochs = range(1, EPOCHS + 1)

# Training Loss
plt.figure()
for wd in WEIGHT_DECAYS:
    plt.plot(epochs, results[wd]["train_losses"], label=f"WD={wd}")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.savefig(os.path.join(REPORTS_DIR, "weight_decay_training_loss.png"))
plt.close()

# Validation Loss
plt.figure()
for wd in WEIGHT_DECAYS:
    plt.plot(epochs, results[wd]["val_losses"], label=f"WD={wd}")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Epoch")
plt.legend()
plt.savefig(os.path.join(REPORTS_DIR, "weight_decay_validation_loss.png"))
plt.close()

# Validation Accuracy
plt.figure()
for wd in WEIGHT_DECAYS:
    plt.plot(epochs, results[wd]["val_accuracies"], label=f"WD={wd}")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs Epoch")
plt.legend()
plt.savefig(os.path.join(REPORTS_DIR, "weight_decay_validation_accuracy.png"))
plt.close()

# Weight Norms
plt.figure()
wds = [str(wd) for wd in WEIGHT_DECAYS]
norms = [results[wd]["weight_norm"] for wd in WEIGHT_DECAYS]
plt.bar(wds, norms)
plt.xlabel("Weight Decay")
plt.ylabel("Final Weight Norm")
plt.title("Effect of Weight Decay on Weight Norm")
plt.savefig(os.path.join(REPORTS_DIR, "weight_norms.png"))
plt.close()

print("\nAll experiments completed.")
print(f"Plots saved to: {REPORTS_DIR}")
