#!/usr/bin/env python3
"""Gamma/hadron classifier: Simple CNN on matrices + MLP on features."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class GammaDataset(Dataset):
    def __init__(self, split: str, feat_mean=None, feat_std=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Matrix: (16, 16, 2) -> transpose to (2, 16, 16)
        mat = self.matrices[idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1))

        # Features: (5,)
        feat = self.features[idx].astype(np.float32)
        if self.feat_mean is not None:
            feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)

        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


def compute_feature_stats(dataset, n_samples=100_000):
    """Compute mean/std for scalar features."""
    rng = np.random.default_rng(42)
    n = len(dataset)
    indices = rng.choice(n, size=min(n_samples, n), replace=False)
    features_list = []
    for idx in indices:
        feat = dataset.features[idx].astype(np.float32)
        features_list.append(feat)
    features = np.stack(features_list)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class CNNClassifier(nn.Module):
    """CNN on 16x16x2 matrices + MLP on features."""

    def __init__(self):
        super().__init__()

        # CNN: process 2x16x16 matrices
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),  # 4x4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )

        # Feature MLP
        self.feat_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, mat, feat):
        x_cnn = self.cnn(mat).view(mat.size(0), -1)  # (batch, 128)
        x_feat = self.feat_mlp(feat)  # (batch, 32)
        x_fusion = torch.cat([x_cnn, x_feat], dim=1)
        logits = self.fusion(x_fusion)
        return logits


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Computing feature statistics...")
    raw_train = GammaDataset("train")
    feat_mean, feat_std = compute_feature_stats(raw_train)

    train_ds = GammaDataset("train", feat_mean=feat_mean, feat_std=feat_std)
    train_loader = DataLoader(
        train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True
    )

    test_ds = GammaDataset("test", feat_mean=feat_mean, feat_std=feat_std)
    test_loader = DataLoader(
        test_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True
    )

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    model = CNNClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Class weights
    labels_all = raw_train.labels[:]
    n_gamma = (labels_all == 0).sum()
    n_hadron = (labels_all == 1).sum()
    w_gamma = len(labels_all) / (2 * n_gamma)
    w_hadron = len(labels_all) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    output_dir = "submissions/haiku-gamma-mar8"
    n_epochs = 40
    best_survival = 1.0

    print(f"Training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for mat, feat, y in train_loader:
            mat, feat, y = mat.to(device), feat.to(device), y.to(device)
            logits = model(mat, feat)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        scheduler.step()

        # Evaluate
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for mat, feat, y in test_loader:
                mat, feat = mat.to(device), feat.to(device)
                logits = model(mat, feat)
                probs = torch.softmax(logits, dim=1)
                gamma_scores = probs[:, 0]
                all_scores.append(gamma_scores.cpu().numpy())
                all_labels.append(y.numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        # Compute survival at 99% gamma efficiency
        is_gamma = labels == 0
        is_hadron = labels == 1
        sg = np.sort(scores[is_gamma])
        ng = len(sg)
        thr_99 = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
        n_hadron_surviving = (scores[is_hadron] >= thr_99).sum()
        survival_99 = n_hadron_surviving / is_hadron.sum()

        lr = optimizer.param_groups[0]["lr"]
        status = ""
        if survival_99 < best_survival:
            best_survival = survival_99
            best_scores = scores
            status = " <- BEST"

        print(
            f"Epoch {epoch+1:2d}/{n_epochs}: "
            f"loss={total_loss/len(train_ds):.4f} "
            f"survival@99={survival_99:.2e} lr={lr:.2e}{status}"
        )

    print(f"\nBest survival @ 99% gamma eff: {best_survival:.2e}")
    np.savez(
        f"{output_dir}/predictions_cnn.npz",
        gamma_scores=best_scores,
    )
    print(f"Saved ({len(best_scores)} scores)")


if __name__ == "__main__":
    main()
