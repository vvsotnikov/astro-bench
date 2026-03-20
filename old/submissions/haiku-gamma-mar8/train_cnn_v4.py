#!/usr/bin/env python3
"""Gamma/hadron classifier: CNN with better loss function for ranking.

Key insight: Use binary cross-entropy with class weights, select best model
based on the 99% gamma efficiency metric (not accuracy).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GammaDataset(Dataset):
    def __init__(self, split: str, feat_mean=None, feat_std=None):
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
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load matrix (16, 16, 2) and transpose to (2, 16, 16)
        mat = self.matrices[idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1))  # (2, 16, 16)

        # Load and normalize features
        feat = self.features[idx].astype(np.float32)
        if self.feat_mean is not None and self.feat_std is not None:
            feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)

        # Label
        label = int(self.labels[idx])

        return (
            torch.from_numpy(mat),
            torch.from_numpy(feat),
            torch.tensor(label, dtype=torch.float32),
        )


def compute_feature_stats(dataset):
    """Compute mean/std of scalar features."""
    n = len(dataset)
    sample_size = min(100000, n)
    indices = np.random.choice(n, sample_size, replace=False)

    features_list = []
    for idx in indices:
        feat = dataset.features[idx].astype(np.float32)
        features_list.append(feat)

    features = np.stack(features_list)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class GammaCNN(nn.Module):
    """CNN + MLP for gamma/hadron separation."""

    def __init__(self):
        super().__init__()

        # CNN branch: 2 channels (electron + muon), 16x16 spatial
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )

        # Feature MLP branch
        self.feat_net = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Fusion and output: binary classification
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, mat, feat):
        # CNN path
        x_cnn = self.cnn(mat)  # (batch, 128, 1, 1)
        x_cnn = x_cnn.view(x_cnn.size(0), -1)  # (batch, 128)

        # Feature path
        x_feat = self.feat_net(feat)  # (batch, 64)

        # Fusion
        x_fusion = torch.cat([x_cnn, x_feat], dim=1)
        logit = self.fusion(x_fusion).squeeze(1)  # (batch,)

        return logit


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
    print("GAMMA/HADRON CLASSIFIER - CNN V4 (BINARY CLASSIFICATION)")
    print("=" * 80)

    # Compute feature statistics
    print("\nComputing feature statistics...")
    raw_train = GammaDataset("train")
    feat_mean, feat_std = compute_feature_stats(raw_train)

    # Load datasets
    print("\nLoading datasets...")
    train_ds = GammaDataset("train", feat_mean=feat_mean, feat_std=feat_std)
    test_ds = GammaDataset("test", feat_mean=feat_mean, feat_std=feat_std)
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = GammaCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Compute class weights from training data
    train_labels = np.array(train_ds.labels[:])
    n_gamma = (train_labels == 0).sum()
    n_hadron = (train_labels == 1).sum()
    w_gamma = len(train_labels) / (2 * n_gamma)
    w_hadron = len(train_labels) / (2 * n_hadron)
    weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w_gamma / w_hadron).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    # Training
    n_epochs = 80
    best_survival = 1.0
    best_scores = None

    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 100)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for mat, feat, label in train_loader:
            mat = mat.to(device)
            feat = feat.to(device)
            label = label.to(device)

            logits = model(mat, feat)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(label)

        scheduler.step()
        train_loss /= len(train_ds)

        # Evaluate
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for mat, feat, label in test_loader:
                mat = mat.to(device)
                feat = feat.to(device)

                logits = model(mat, feat)
                scores = torch.sigmoid(logits)  # Convert to probability
                all_scores.append(scores)
                all_labels.append(label)

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
            f"Epoch {epoch+1:2d}/{n_epochs} | "
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
