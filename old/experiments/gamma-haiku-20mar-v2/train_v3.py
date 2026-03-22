"""v3: Dual-channel CNN (electron + muon) with attention pooling.

Use both channels. Attention helps focus on discriminative regions.
Gammas: electron-dominated, muon-sparse.
Hadrons: muon-enriched, electron-dominated.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_train, load_test
from verify import evaluate


class AttentionPool(nn.Module):
    """Learnable attention pooling over spatial dimensions."""

    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (N, C, H, W)
        attn = torch.sigmoid(self.attn(x))  # (N, 1, H, W)
        weighted = x * attn
        return weighted.sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)


class DualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared encoder
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.attn = AttentionPool(64)

        # Fully connected
        self.mlp = nn.Sequential(
            nn.Linear(64 + 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x, scalar):
        # x: (N, 16, 16, 2) from loader
        x = x.permute(0, 3, 1, 2)  # (N, 2, 16, 16)
        feat = self.conv(x)  # (N, 64, 16, 16)
        feat = self.attn(feat)  # (N, 64)
        x = torch.cat([feat, scalar], dim=1)
        return self.mlp(x)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

    # Normalize matrices
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    x_mean = X_train.mean()
    x_std = X_train.std()
    X_train = (X_train - x_mean) / (x_std + 1e-6)
    X_test = (X_test - x_mean) / (x_std + 1e-6)

    # Scalar features
    scalar_train = np.stack([
        f_train[:, 3] - f_train[:, 4],  # log(Ne) - log(Nmu)
        f_train[:, 1] / 30,  # Zenith
        f_train[:, 0] / 18,  # Energy (log10(eV))
        np.log1p(f_train[:, 3]),  # log(1 + Ne)
    ], axis=1).astype(np.float32)

    scalar_test = np.stack([
        f_test[:, 3] - f_test[:, 4],
        f_test[:, 1] / 30,
        f_test[:, 0] / 18,
        np.log1p(f_test[:, 3]),
    ], axis=1).astype(np.float32)

    s_mean = scalar_train.mean(axis=0)
    s_std = scalar_train.std(axis=0)
    scalar_train = (scalar_train - s_mean) / (s_std + 1e-6)
    scalar_test = (scalar_test - s_mean) / (s_std + 1e-6)

    # Train/val split
    rng = np.random.default_rng(42)
    n = len(y_train)
    val_mask = rng.random(n) < 0.2
    train_mask = ~val_mask

    train_x = torch.from_numpy(X_train[train_mask])
    train_s = torch.from_numpy(scalar_train[train_mask])
    train_y = torch.from_numpy(y_train[train_mask]).long()

    val_x = torch.from_numpy(X_train[val_mask])
    val_s = torch.from_numpy(scalar_train[val_mask])
    val_y = torch.from_numpy(y_train[val_mask]).long()

    test_x = torch.from_numpy(X_test)
    test_s = torch.from_numpy(scalar_test)
    test_y = torch.from_numpy(y_test).long()

    print(f"  Train: {len(train_y):,}  Val: {len(val_y):,}")

    model = DualCNN().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    class_weights = torch.tensor(
        [len(y_train) / (2 * n_gamma), len(y_train) / (2 * n_hadron)],
        dtype=torch.float32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    print("\nTraining...")
    for epoch in range(35):
        model.train()
        train_loss = 0
        for i in range(0, len(train_y), 4096):
            end = min(i + 4096, len(train_y))
            x = train_x[i:end].to(device)
            s = train_s[i:end].to(device)
            y = train_y[i:end].to(device)
            logits = model(x, s)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * (end - i)

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(val_y), 8192):
                end = min(i + 8192, len(val_y))
                x = val_x[i:end].to(device)
                s = val_s[i:end].to(device)
                y = val_y[i:end].to(device)
                logits = model(x, s)
                val_loss += criterion(logits, y).item() * (end - i)
                all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

        val_loss /= len(val_y)
        val_scores = np.concatenate(all_scores)

        print(f"Epoch {epoch+1:2d}/35: train_loss={train_loss/len(train_y):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = val_scores
            torch.save(model.state_dict(), "model_v3.pt")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_scores = []
    with torch.no_grad():
        for i in range(0, len(test_y), 8192):
            end = min(i + 8192, len(test_y))
            x = test_x[i:end].to(device)
            s = test_s[i:end].to(device)
            logits = model(x, s)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)
    np.savez("predictions_v3.npz", gamma_scores=test_scores)

    metric = evaluate(test_scores, "v3: Dual-channel CNN (e+mu) + attention pooling + scalar features")
    return metric


if __name__ == "__main__":
    main()
