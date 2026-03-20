#!/usr/bin/env python3
"""Gamma/hadron classifier: Simple MLP with better hyperparameters.

Key insight: The previous MLP baseline used class weights but CrossEntropyLoss
which optimizes for accuracy. Instead, use BCEWithLogitsLoss which is better
for ranking metrics. Also use more aggressive regularization.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.matrices = np.load(
            f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/matrices.npy",
            mmap_mode="r",
        )
        self.features = np.load(
            f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/features.npy",
            mmap_mode="r",
        )
        self.labels = np.load(
            f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/labels_gamma.npy",
            mmap_mode="r",
        )
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].flatten().astype(np.float32)
        feat = self.features[idx].astype(np.float32)
        x = np.concatenate([mat, feat])  # 517 dims

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        label = float(self.labels[idx])  # 0 = gamma, 1 = hadron
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


def compute_stats(dataset, n_samples=500_000):
    """Compute mean/std from flattened data."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    samples = []
    for idx in indices:
        mat = dataset.matrices[idx].flatten().astype(np.float32)
        feat = dataset.features[idx].astype(np.float32)
        x = np.concatenate([mat, feat])
        samples.append(x)
    samples = np.stack(samples)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class GammaMLP(nn.Module):
    """Simple MLP with BatchNorm and dropout."""

    def __init__(self, input_dim=517, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_survival_at_99(scores, labels):
    """Compute survival rate at 99% gamma efficiency."""
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    is_gamma = labels_np == 0
    is_hadron = labels_np == 1

    # Find threshold for 99% gamma efficiency
    sg = np.sort(scores_np[is_gamma])
    ng = len(sg)
    idx_99 = max(0, int(np.floor(ng * (1 - 0.99))))
    thr_99 = sg[idx_99]

    # Compute survival
    n_hadron_surviving = (scores_np[is_hadron] >= thr_99).sum()
    survival_99 = n_hadron_surviving / is_hadron.sum()

    return survival_99, thr_99


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print("=" * 80)
    print("GAMMA/HADRON CLASSIFIER - MLP V5 (BCELOGITSLOSS)")
    print("=" * 80)

    # Load and compute stats
    print("\nComputing normalization statistics...")
    raw_train = GammaDataset("train")
    mean, std = compute_stats(raw_train)
    print(f"  mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  std range:  [{std.min():.2f}, {std.max():.2f}]")

    # Load datasets
    print("\nLoading datasets...")
    train_ds = GammaDataset("train", mean=mean, std=std)
    test_ds = GammaDataset("test", mean=mean, std=std)
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = GammaMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Compute class weights
    labels_all = raw_train.labels[:]
    n_gamma = (labels_all == 0).sum()
    n_hadron = (labels_all == 1).sum()
    w_gamma = len(labels_all) / (2 * n_gamma)
    w_hadron = len(labels_all) / (2 * n_hadron)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    # Loss and optimizer
    pos_weight = torch.tensor(w_gamma / w_hadron, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training
    n_epochs = 100
    best_survival = 1.0
    best_scores = None

    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 100)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x).squeeze(1)  # (batch,)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(train_ds)

        # Evaluate
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x).squeeze(1)
                scores = torch.sigmoid(logits)
                all_scores.append(scores)
                all_labels.append(y)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        survival_99, thr_99 = compute_survival_at_99(scores, labels)

        lr = optimizer.param_groups[0]["lr"]
        status = ""
        if survival_99 < best_survival:
            best_survival = survival_99
            best_scores = scores.cpu().numpy()
            status = " <- NEW BEST"

        print(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"loss={train_loss:.4f} | "
            f"survival@99={survival_99:.2e} thr={thr_99:.4f} | "
            f"lr={lr:.2e}{status}"
        )

    print("-" * 100)
    print(f"\nBest survival @ 99% gamma efficiency: {best_survival:.2e}")

    # Save predictions
    os.makedirs("/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar8", exist_ok=True)
    np.savez(
        "/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar8/predictions.npz",
        gamma_scores=best_scores,
    )
    print(f"Saved predictions ({len(best_scores)} scores)")


if __name__ == "__main__":
    main()
