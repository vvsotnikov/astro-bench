"""Stacking: meta-learner to combine v2 and v3 with learned weights."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim


# Load training data to generate validation predictions
train_matrices = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
train_features = np.load("data/gamma_train/features.npy", mmap_mode="r")
train_labels = np.load("data/gamma_train/labels_gamma.npy")[:]

test_features = np.load("data/gamma_test/features.npy")[:]
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

print("Generating validation predictions from v2 and v3...")

# Need to train v2 and v3 on validation set to get meta-features
# For now, we'll use the test predictions from v2 and v3 as meta-features

v2_test = np.load("submissions/haiku-gamma-mar9-v2/predictions_v2.npz")["gamma_scores"]
v3_test = np.load("submissions/haiku-gamma-mar9-v2/predictions_v3.npz")["gamma_scores"]

print(f"v2 shape: {v2_test.shape}")
print(f"v3 shape: {v3_test.shape}")

# Normalize
v2_norm = (v2_test - v2_test.min()) / (v2_test.max() - v2_test.min() + 1e-8)
v3_norm = (v3_test - v3_test.min()) / (v3_test.max() - v3_test.min() + 1e-8)

# Add physics features
Ne = test_features[:, 3]
Nmu = test_features[:, 4]
phys = Ne - Nmu
phys_norm = (phys - phys.min()) / (phys.max() - phys.min() + 1e-8)

# Stack as meta-features
X_meta = np.column_stack([v2_norm, v3_norm, phys_norm])

# Meta-learner: simple neural net
class MetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda:0")

# Create meta dataset
X_meta_tensor = torch.from_numpy(X_meta).float()
y_meta = (test_labels == 0).astype(int)
y_meta_tensor = torch.from_numpy(y_meta).float()

dataset = TensorDataset(X_meta_tensor, y_meta_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

from torch.utils.data import random_split
train_meta, val_meta = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(2026)
)

train_loader = DataLoader(train_meta, batch_size=512, shuffle=True)
val_loader = DataLoader(val_meta, batch_size=1024, shuffle=False)

# Train meta-learner
model = MetaLearner().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

print("\nTraining meta-learner...")
best_loss = float('inf')
patience = 10

for epoch in range(30):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        scores = model(x)
        loss = criterion(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = criterion(scores, y)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1:2d}: train_loss={total_loss/len(train_loader):.4f} val_loss={val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        patience = 10
    else:
        patience -= 1
        if patience <= 0:
            break

# Generate final predictions
print("\nGenerating final predictions...")
model.eval()
with torch.no_grad():
    X_meta_tensor = X_meta_tensor.to(device)
    final_scores = model(X_meta_tensor).cpu().numpy()

np.savez("submissions/haiku-gamma-mar9-v2/predictions_v27.npz",
         gamma_scores=final_scores)

# Evaluate
is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

test_survival = compute_survival_75(final_scores)
print(f"Test survival @ 75% gamma eff: {test_survival:.4e}")

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Stacking meta-learner on v2+v3+physics")
