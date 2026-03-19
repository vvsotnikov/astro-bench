#!/usr/bin/env python3
"""Gamma/hadron v16: Vision Transformer on detector matrices.

Different inductive bias than CNN: self-attention over patches instead of local convolutions.
Learns global relationships in detector patterns.
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


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=16, patch_size=4, in_channels=2, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        # x: (B, 2, 16, 16)
        x = self.proj(x)  # (B, embed_dim, 4, 4)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x


class ViTBranch(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=16, patch_size=4, in_channels=2, embed_dim=embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=256, dropout=0.1)
              for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_dim, 64)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, n_patches+1, embed_dim)
        x = self.transformer(x)
        x = x[:, 0]  # Take CLS token
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


class ViTFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_branch = ViTBranch(embed_dim=128, num_heads=4, num_layers=3)
        self.feat_branch = FeatureBranch()
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, matrices, features):
        vit_out = self.vit_branch(matrices)
        feat_out = self.feat_branch(features)
        fused = torch.cat([vit_out, feat_out], dim=1)
        return self.fusion(fused)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Loading data...")
    raw_train = GammaDataset("train")
    print(f"  Training set size: {len(raw_train)}")

    print("Computing normalization stats...")
    mean_mat, std_mat, mean_feat, std_feat = compute_stats(raw_train)

    train_ds = GammaDataset("train", mean_mat=mean_mat, std_mat=std_mat,
                           mean_feat=mean_feat, std_feat=std_feat)
    test_ds = GammaDataset("test", mean_mat=mean_mat, std_mat=std_mat,
                          mean_feat=mean_feat, std_feat=std_feat)

    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=4)

    model = ViTFusion().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    labels_all = raw_train.labels[:]
    n_gamma = (labels_all == 0).sum()
    n_hadron = (labels_all == 1).sum()
    w_gamma = len(labels_all) / (2 * n_gamma)
    w_hadron = len(labels_all) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

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
            torch.save(model.state_dict(), "submissions/run1/model_v16.pt")

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("submissions/run1/model_v16.pt"))
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for mat, feat, y in test_loader:
            mat, feat, y = mat.to(device), feat.to(device), y.to(device)
            logits = model(mat, feat)
            probs = torch.softmax(logits, dim=1)
            gamma_scores = probs[:, 0]
            all_scores.append(gamma_scores.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    gamma_scores = np.concatenate(all_scores)
    test_labels = np.concatenate(all_labels)

    os.makedirs("submissions/run1", exist_ok=True)
    np.savez("submissions/run1/predictions_v16.npz", gamma_scores=gamma_scores)
    np.save("submissions/run1/probs_v16.npy", gamma_scores)

    is_gamma = test_labels == 0
    is_hadron = test_labels == 1
    sg = np.sort(gamma_scores[is_gamma])
    ng = len(sg)
    thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (gamma_scores[is_hadron] >= thr_75).sum()
    survival_75 = n_hadron_surviving / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

    print(f"\nSurvival rate @ 75% gamma efficiency: {survival_75:.2e}")
    print(f"Saved predictions to submissions/run1/predictions_v16.npz")


if __name__ == "__main__":
    main()
