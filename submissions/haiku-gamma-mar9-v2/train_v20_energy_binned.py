"""Energy-dependent models: separate regression DNN for each energy bin."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


class GammaDataset(Dataset):
    def __init__(self, split: str, indices=None, mean=None, std=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.indices = indices if indices is not None else np.arange(len(self.labels))
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        mat = self.matrices[actual_idx].flatten().astype(np.float32)
        feat = self.features[actual_idx].astype(np.float32)
        x = np.concatenate([mat, feat])
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        label = int(self.labels[actual_idx])
        return torch.from_numpy(x), label


class RegDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(517, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_stats(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def compute_survival_at_75(gamma_scores, labels):
    if len(gamma_scores) == 0 or len(labels) == 0:
        return 1.0
    is_gamma = labels == 0
    is_hadron = labels == 1
    if is_gamma.sum() == 0 or is_hadron.sum() == 0:
        return 1.0
    sg = np.sort(gamma_scores[is_gamma])
    ng = len(sg)
    if ng == 0:
        return 1.0
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (gamma_scores[is_hadron] >= thr).sum()
    return n_hadron_surviving / is_hadron.sum()


device = torch.device("cuda:0")
print(f"Device: {device}\n")

# Load all data
print("Loading data...")
train_matrices = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
train_features = np.load("data/gamma_train/features.npy", mmap_mode="r")
train_labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

test_matrices = np.load("data/gamma_test/matrices.npy", mmap_mode="r")
test_features = np.load("data/gamma_test/features.npy", mmap_mode="r")
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

# Get energy bins
E_train = train_features[:, 0]
E_test = test_features[:, 0]

# Define energy bins
energy_bins = [(14.0, 15.0), (15.0, 15.5), (15.5, 16.0), (16.0, 16.5), (16.5, 17.0), (17.0, 18.0)]
print(f"Training energy-binned models for {len(energy_bins)} bins\n")

# Store test predictions
all_test_scores = np.zeros(len(test_labels))
n_train_total = len(train_labels)

for bin_idx, (E_min, E_max) in enumerate(energy_bins):
    print(f"=== Bin {bin_idx+1}: E ∈ [{E_min}, {E_max}) ===")

    # Get indices for this bin
    train_mask = (E_train >= E_min) & (E_train < E_max)
    test_mask = (E_test >= E_min) & (E_test < E_max)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    n_train_bin = len(train_indices)
    n_test_bin = len(test_indices)

    print(f"Train: {n_train_bin}, Test: {n_test_bin}")

    if n_train_bin < 100 or n_test_bin < 10:
        print("Skip: too few samples\n")
        continue

    # Create dataset
    train_ds_full = GammaDataset("train", indices=train_indices)

    # Compute stats on this bin
    mat_flat = np.array([train_matrices[i].flatten() for i in train_indices]).astype(np.float32)
    feat_part = np.array([train_features[i] for i in train_indices]).astype(np.float32)
    X_bin = np.concatenate([mat_flat, feat_part], axis=1)
    mean, std = compute_stats(X_bin)

    # Create normalized dataset
    class NormDataset(GammaDataset):
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            mat = train_matrices[actual_idx].flatten().astype(np.float32)
            feat = train_features[actual_idx].astype(np.float32)
            x = np.concatenate([mat, feat])
            x = (x - mean) / (std + 1e-8)
            label = int(train_labels[actual_idx])
            return torch.from_numpy(x), label

    train_ds = NormDataset("train", indices=train_indices)

    # Train/val split
    n_train_split = int(0.8 * len(train_indices))
    val_indices_bin = train_indices[n_train_split:]
    train_indices_bin = train_indices[:n_train_split]

    train_loader = DataLoader(
        NormDataset("train", indices=train_indices_bin),
        batch_size=1024, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        NormDataset("train", indices=val_indices_bin),
        batch_size=2048, shuffle=False, num_workers=4, pin_memory=True
    )

    # Train model
    model = RegDNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_survival = 1.0
    patience = 10

    for epoch in range(30):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            target = (y == 0).float()
            scores = model(x)
            loss = criterion(scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            all_scores = []
            all_labels = []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    scores = model(x).cpu().numpy()
                    all_scores.append(scores)
                    all_labels.append(y.numpy())

            val_scores = np.concatenate(all_scores)
            val_labels = np.concatenate(all_labels)
            val_survival = compute_survival_at_75(val_scores, val_labels)

            if epoch > 0:
                if val_survival < best_survival:
                    best_survival = val_survival
                    patience = 10
                else:
                    patience -= 1
                    if patience <= 0:
                        break

    # Test inference
    model.eval()
    test_bin_matrices = test_matrices[test_indices].reshape(len(test_indices), -1).astype(np.float32)
    test_bin_features = test_features[test_indices].astype(np.float32)
    test_bin_X = np.concatenate([test_bin_matrices, test_bin_features], axis=1)
    test_bin_X_norm = (test_bin_X - mean) / (std + 1e-8)

    test_tensor = torch.from_numpy(test_bin_X_norm).to(device)
    with torch.no_grad():
        bin_scores = model(test_tensor).cpu().numpy()

    all_test_scores[test_indices] = bin_scores
    print(f"Test scores for bin: {compute_survival_at_75(bin_scores, test_labels[test_indices]):.4e}\n")

# Save
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v20.npz",
         gamma_scores=all_test_scores)

# Evaluate
test_survival = compute_survival_at_75(all_test_scores, test_labels)
print(f"Overall test survival @ 75% gamma eff: {test_survival:.4e}")

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Energy-binned regression models")
