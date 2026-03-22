#!/usr/bin/env python3
"""Gamma/hadron v1: Simple DNN on flattened matrices + features.

Baseline approach: binary classification with cross-entropy loss and class weighting.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import os

class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].flatten().astype(np.float32)
        feat = self.features[idx].astype(np.float32)
        x = np.concatenate([mat, feat])  # 517 dims
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(x), int(self.labels[idx])


def compute_stats(dataset, n_samples=200_000):
    """Compute mean/std for normalization."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    samples = []
    for idx in indices:
        mat = dataset.matrices[idx].flatten().astype(np.float32)
        feat = dataset.features[idx].astype(np.float32)
        samples.append(np.concatenate([mat, feat]))
    samples = np.stack(samples)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class DNN(nn.Module):
    def __init__(self, input_dim=517, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    raw_train = GammaDataset("train")
    print(f"  Training set size: {len(raw_train)}")

    # Compute normalization stats
    print("Computing normalization stats...")
    mean, std = compute_stats(raw_train)
    print(f"  mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  std range:  [{std.min():.2f}, {std.max():.2f}]")

    # Create datasets
    train_ds = GammaDataset("train", mean=mean, std=std)
    test_ds = GammaDataset("test", mean=mean, std=std)

    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4)

    # Model
    model = DNN(hidden=256, dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Class weights (gamma is ~5% in training)
    labels_all = raw_train.labels[:]
    n_gamma = (labels_all == 0).sum()
    n_hadron = (labels_all == 1).sum()
    w_gamma = len(labels_all) / (2 * n_gamma)
    w_hadron = len(labels_all) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    best_test_scores = None
    n_epochs = 20

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
        val_loss /= len(val_ds)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}/{n_epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "submissions/run1/model_v1.pt")

    # Evaluate on test set using best model
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("submissions/run1/model_v1.pt"))
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            gamma_scores = probs[:, 0]  # P(gamma)
            all_scores.append(gamma_scores.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    gamma_scores = np.concatenate(all_scores)
    test_labels = np.concatenate(all_labels)

    # Save predictions
    os.makedirs("submissions/run1", exist_ok=True)
    np.savez("submissions/run1/predictions_v1.npz", gamma_scores=gamma_scores)
    np.save("submissions/run1/probs_v1.npy", gamma_scores)

    # Compute survival @ 75% gamma efficiency for reference
    is_gamma = test_labels == 0
    is_hadron = test_labels == 1
    sg = np.sort(gamma_scores[is_gamma])
    ng = len(sg)
    thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (gamma_scores[is_hadron] >= thr_75).sum()
    survival_75 = n_hadron_surviving / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

    print(f"\nSurvival rate @ 75% gamma efficiency: {survival_75:.2e}")
    print(f"Saved predictions to submissions/run1/predictions_v1.npz")


if __name__ == "__main__":
    main()
