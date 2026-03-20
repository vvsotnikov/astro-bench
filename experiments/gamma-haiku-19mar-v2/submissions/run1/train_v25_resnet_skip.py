#!/usr/bin/env python3
"""Gamma/hadron v25: Proper ResNet with actual skip connections on matrices.

v4 was supposed to be a ResNet but didn't have proper skip connections.
This is a true residual network with skip connections at each stage.
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
        mat = self.matrices[idx].astype(np.float32)
        feat = self.features[idx].astype(np.float32)
        if self.mean_mat is not None:
            mat = (mat - self.mean_mat) / (self.std_mat + 1e-8)
        if self.mean_feat is not None:
            feat = (feat - self.mean_feat) / (self.std_feat + 1e-8)
        mat = np.transpose(mat, (2, 0, 1))
        return torch.from_numpy(mat), torch.from_numpy(feat), int(self.labels[idx])


def compute_stats(dataset, n_samples=200_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    mat_samples = []
    feat_samples = []
    for idx in indices:
        mat_samples.append(dataset.matrices[idx].astype(np.float32))
        feat_samples.append(dataset.features[idx].astype(np.float32))
    mat_samples = np.stack(mat_samples)
    feat_samples = np.stack(feat_samples)
    mean_mat = mat_samples.mean(axis=(0, 1, 2))
    std_mat = mat_samples.std(axis=(0, 1, 2))
    std_mat[std_mat == 0] = 1.0
    mean_feat = feat_samples.mean(axis=0)
    std_feat = feat_samples.std(axis=0)
    std_feat[std_feat == 0] = 1.0
    return mean_mat, std_mat, mean_feat, std_feat


class ResidualBlock(nn.Module):
    """Residual block with option for stride/channel change."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut: if stride > 1 or channels change, use 1x1 convolution
        self.shortcut = nn.Sequential()
        if stride > 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # residual connection
        out = self.relu(out)
        return out


class ResNetBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class FeatureBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class ResNetFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_branch = ResNetBranch()
        self.feat_branch = FeatureBranch()
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, matrices, features):
        cnn_out = self.resnet_branch(matrices)
        feat_out = self.feat_branch(features)
        fused = torch.cat([cnn_out, feat_out], dim=1)
        return self.fusion(fused)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    raw_train = GammaDataset("train")
    print(f"Training set size: {len(raw_train)}")

    mean_mat, std_mat, mean_feat, std_feat = compute_stats(raw_train)

    train_ds = GammaDataset("train", mean_mat=mean_mat, std_mat=std_mat,
                           mean_feat=mean_feat, std_feat=std_feat)
    test_ds = GammaDataset("test", mean_mat=mean_mat, std_mat=std_mat,
                          mean_feat=mean_feat, std_feat=std_feat)

    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=4)

    torch.manual_seed(42)
    model = ResNetFusion().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    labels_all = raw_train.labels[:].astype(np.float32)
    n_gamma = (labels_all == 0).sum()
    n_hadron = (labels_all == 1).sum()
    w_gamma = len(labels_all) / (2 * n_gamma)
    w_hadron = len(labels_all) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    n_epochs = 20

    for epoch in range(n_epochs):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "submissions/run1/model_v25.pt")

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("submissions/run1/model_v25.pt"))
    model.eval()
    all_scores = []
    with torch.no_grad():
        for mat, feat, y in test_loader:
            mat, feat, y = mat.to(device), feat.to(device), y.to(device)
            logits = model(mat, feat)
            probs = torch.softmax(logits, dim=1)
            gamma_scores = probs[:, 0]
            all_scores.append(gamma_scores.cpu().numpy())

    gamma_scores = np.concatenate(all_scores)

    os.makedirs("submissions/run1", exist_ok=True)
    np.save("submissions/run1/probs_v25.npy", gamma_scores)

    print(f"Saved v25 scores")


if __name__ == "__main__":
    main()
