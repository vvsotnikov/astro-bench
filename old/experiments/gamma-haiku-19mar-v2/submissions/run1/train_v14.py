#!/usr/bin/env python3
"""Gamma/hadron v14: CNN with attention, slightly larger model.

Increase channels: 32->48->96, larger hidden dims in fusion.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import os

class GammaDataset(Dataset):
    def __init__(self, split: str, mean_mat=None, std_mat=None, mean_feat=None, std_feat=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean_mat = mean_mat
        self.std_mat = std_mat
        self.mean_feat = mean_feat
        self.std_feat = std_feat

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        feat = self.features[idx].astype(np.float32)  # (5,)

        if self.mean_mat is not None:
            mat = (mat - self.mean_mat) / (self.std_mat + 1e-8)
        if self.mean_feat is not None:
            feat = (feat - self.mean_feat) / (self.std_feat + 1e-8)

        mat = np.transpose(mat, (2, 0, 1))  # (2, 16, 16)
        return torch.from_numpy(mat), torch.from_numpy(feat), int(self.labels[idx])


def compute_stats(dataset, n_samples=200_000):
    """Compute mean/std for normalization (per channel)."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mat_samples = []
    feat_samples = []
    for idx in indices:
        mat_samples.append(dataset.matrices[idx].astype(np.float32))
        feat_samples.append(dataset.features[idx].astype(np.float32))

    mat_samples = np.stack(mat_samples)  # (N, 16, 16, 2)
    feat_samples = np.stack(feat_samples)  # (N, 5)

    # Per-channel statistics for matrices
    mean_mat = mat_samples.mean(axis=(0, 1, 2))  # (2,)
    std_mat = mat_samples.std(axis=(0, 1, 2))   # (2,)
    std_mat[std_mat == 0] = 1.0

    mean_feat = feat_samples.mean(axis=0)  # (5,)
    std_feat = feat_samples.std(axis=0)    # (5,)
    std_feat[std_feat == 0] = 1.0

    return mean_mat, std_mat, mean_feat, std_feat


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, max(in_ch // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_ch // reduction, 4), in_ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class CNNAttentionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.att1 = ChannelAttention(48)

        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),  # 16->8
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.att2 = ChannelAttention(96)

        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),  # 8->4
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.att3 = ChannelAttention(192)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.att1(x)

        x = self.conv2(x)
        x = self.att2(x)

        x = self.conv3(x)
        x = self.att3(x)

        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class FeatureBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class CNNAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_branch = CNNAttentionBranch()
        self.feat_branch = FeatureBranch()
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, matrices, features):
        cnn_out = self.cnn_branch(matrices)
        feat_out = self.feat_branch(features)
        fused = torch.cat([cnn_out, feat_out], dim=1)
        return self.fusion(fused)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    raw_train = GammaDataset("train")
    print(f"  Training set size: {len(raw_train)}")

    # Compute normalization stats
    print("Computing normalization stats...")
    mean_mat, std_mat, mean_feat, std_feat = compute_stats(raw_train)

    # Create datasets
    train_ds = GammaDataset("train", mean_mat=mean_mat, std_mat=std_mat,
                           mean_feat=mean_feat, std_feat=std_feat)
    test_ds = GammaDataset("test", mean_mat=mean_mat, std_mat=std_mat,
                          mean_feat=mean_feat, std_feat=std_feat)

    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=4)

    # Model
    model = CNNAttentionFusion().to(device)
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

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    n_epochs = 20

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for mat, feat, y in train_loader:
            mat, feat, y = mat.to(device), feat.to(device), y.to(device)
            logits = model(mat, feat)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mat, feat, y in val_loader:
                mat, feat, y = mat.to(device), feat.to(device), y.to(device)
                logits = model(mat, feat)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
        val_loss /= len(val_ds.dataset)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}/{n_epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "submissions/run1/model_v14.pt")

    # Evaluate on test set using best model
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("submissions/run1/model_v14.pt"))
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for mat, feat, y in test_loader:
            mat, feat, y = mat.to(device), feat.to(device), y.to(device)
            logits = model(mat, feat)
            probs = torch.softmax(logits, dim=1)
            gamma_scores = probs[:, 0]  # P(gamma)
            all_scores.append(gamma_scores.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    gamma_scores = np.concatenate(all_scores)
    test_labels = np.concatenate(all_labels)

    # Save predictions
    os.makedirs("submissions/run1", exist_ok=True)
    np.savez("submissions/run1/predictions_v14.npz", gamma_scores=gamma_scores)
    np.save("submissions/run1/probs_v14.npy", gamma_scores)

    # Compute survival @ 75% gamma efficiency for reference
    is_gamma = test_labels == 0
    is_hadron = test_labels == 1
    sg = np.sort(gamma_scores[is_gamma])
    ng = len(sg)
    thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (gamma_scores[is_hadron] >= thr_75).sum()
    survival_75 = n_hadron_surviving / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

    print(f"\nSurvival rate @ 75% gamma efficiency: {survival_75:.2e}")
    print(f"Saved predictions to submissions/run1/predictions_v14.npz")


if __name__ == "__main__":
    main()
