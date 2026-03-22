"""v10: Larger dual-channel CNN variant (more capacity, more epochs).

v3 was successful with dual channels + attention.
Try larger network with more training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_train, load_test
from verify import evaluate


class AttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        attn = torch.sigmoid(self.attn(x))
        weighted = x * attn
        return weighted.sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)


class LargerDualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.attn = AttentionPool(128)

        self.mlp = nn.Sequential(
            nn.Linear(128 + 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
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
        x = x.permute(0, 3, 1, 2)  # (N, 16, 16, 2) → (N, 2, 16, 16)
        feat = self.conv(x)
        feat = self.attn(feat)
        x = torch.cat([feat, scalar], dim=1)
        return self.mlp(x)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

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
        f_train[:, 3] - f_train[:, 4],  # Ne - Nmu
        f_train[:, 1] / 30,  # Zenith
        f_train[:, 0] / 18,  # Energy
        np.log1p(f_train[:, 3]),  # log(Ne)
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

    train_loader = DataLoader(TensorDataset(train_x, train_s, train_y), batch_size=4096, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_s, val_y), batch_size=8192, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_s, test_y), batch_size=8192, shuffle=False)

    model = LargerDualCNN().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    class_weights = torch.tensor(
        [len(y_train) / (2 * n_gamma), len(y_train) / (2 * n_hadron)],
        dtype=torch.float32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    print("\nTraining...")
    for epoch in range(50):
        model.train()
        train_loss = 0
        for x, s, y in train_loader:
            x, s, y = x.to(device), s.to(device), y.to(device)
            logits = model(x, s)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        scheduler.step()

        model.eval()
        val_loss = 0
        all_scores = []
        with torch.no_grad():
            for x, s, y in val_loader:
                x, s, y = x.to(device), s.to(device), y.to(device)
                logits = model(x, s)
                val_loss += criterion(logits, y).item() * len(y)
                all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

        val_loss /= len(val_y)
        val_scores = np.concatenate(all_scores)

        if epoch % 10 == 0 or epoch == 49:
            print(f"Epoch {epoch+1:2d}/50: train_loss={train_loss/len(train_y):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = val_scores
            torch.save(model.state_dict(), "model_v10.pt")

    print("\nEvaluating on test set...")
    model.eval()
    test_scores = []
    with torch.no_grad():
        for x, s, y in test_loader:
            x, s = x.to(device), s.to(device)
            logits = model(x, s)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)
    np.savez("predictions_v10.npz", gamma_scores=test_scores)

    metric = evaluate(test_scores, "v10: Larger dual-channel CNN (96-128 channels, 50 epochs, more capacity)")
    return metric


if __name__ == "__main__":
    main()
