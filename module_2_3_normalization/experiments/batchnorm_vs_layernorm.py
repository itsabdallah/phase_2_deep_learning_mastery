import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Dataset
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# =========================
# Model Definition
# =========================
class MLP(nn.Module):
    def __init__(self, norm_type="batch"):
        super().__init__()

        def norm_layer(dim):
            if norm_type == "batch":
                return nn.BatchNorm1d(dim)
            elif norm_type == "layer":
                return nn.LayerNorm(dim)
            else:
                return nn.Identity()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            norm_layer(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            norm_layer(128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Training Loop
# =========================
def train_model(norm_type, batch_size, epochs=10, lr=1e-3):
    set_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MLP(norm_type=norm_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"[{norm_type.upper()} | BS={batch_size}] "
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return train_losses, val_losses, val_accuracies

# =========================
# Experiments
# =========================
results = {}

configs = [
    ("batch", 128),
    ("batch", 16),
    ("layer", 128),
    ("layer", 16),
]

for norm_type, batch_size in configs:
    key = f"{norm_type}_bs{batch_size}"
    results[key] = train_model(norm_type, batch_size)

# =========================
# Plotting
# =========================
reports_dir = Path(__file__).resolve().parents[1] / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

epochs = range(1, 11)

# ---- Training Loss ----
plt.figure()
for key, (train_losses, _, _) in results.items():
    plt.plot(epochs, train_losses, label=key)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("BatchNorm vs LayerNorm — Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(reports_dir / "batchnorm_vs_layernorm_training_loss.png")
plt.close()

# ---- Validation Accuracy ----
plt.figure()
for key, (_, _, val_acc) in results.items():
    plt.plot(epochs, val_acc, label=key)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("BatchNorm vs LayerNorm — Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(reports_dir / "batchnorm_vs_layernorm_validation_accuracy.png")
plt.close()

print(f"Saved plots to: {reports_dir}")
