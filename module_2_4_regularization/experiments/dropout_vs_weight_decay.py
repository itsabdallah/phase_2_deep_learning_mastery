import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# Configuration
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01
WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.5

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
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
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------------------------
# Model Definitions
# -------------------------
class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


class WeightDecayNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Training / Evaluation
# -------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total

# -------------------------
# Plot Helper (CRITICAL)
# -------------------------
def save_plot(epochs, ys, labels, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    for y, label in zip(ys, labels):
        plt.plot(epochs, y, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -------------------------
# Experiment Runner
# -------------------------
def run_experiment(model, optimizer):
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return train_losses, val_losses, val_accs

# -------------------------
# Main
# -------------------------
print(f"Using device: {DEVICE}")

epochs = list(range(1, EPOCHS + 1))

# Dropout experiment
dropout_model = DropoutNet().to(DEVICE)
dropout_optimizer = optim.SGD(dropout_model.parameters(), lr=LR)

d_train, d_val, d_acc = run_experiment(dropout_model, dropout_optimizer)

# Weight decay experiment
wd_model = WeightDecayNet().to(DEVICE)
wd_optimizer = optim.SGD(
    wd_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)

w_train, w_val, w_acc = run_experiment(wd_model, wd_optimizer)

# -------------------------
# Save Plots
# -------------------------
save_plot(
    epochs,
    [d_train, w_train],
    ["Dropout", "Weight Decay"],
    "Dropout vs Weight Decay — Training Loss",
    "Loss",
    os.path.join(REPORT_DIR, "dropout_vs_weight_decay_training_loss.png")
)

save_plot(
    epochs,
    [d_val, w_val],
    ["Dropout", "Weight Decay"],
    "Dropout vs Weight Decay — Validation Loss",
    "Loss",
    os.path.join(REPORT_DIR, "dropout_vs_weight_decay_validation_loss.png")
)

save_plot(
    epochs,
    [d_acc, w_acc],
    ["Dropout", "Weight Decay"],
    "Dropout vs Weight Decay — Validation Accuracy",
    "Accuracy",
    os.path.join(REPORT_DIR, "dropout_vs_weight_decay_validation_accuracy.png")
)

print(f"Plots saved to: {REPORT_DIR}")
