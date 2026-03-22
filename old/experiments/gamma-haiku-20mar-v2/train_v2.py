"""v2: CNN on muon channel + scalar features.

Physics: Muon channel is the key discriminator. Gammas: Nmu ≈ 3.0, Hadrons: Nmu ≈ 3.5.
Use muon spatial information as primary input, augment with Ne/Nmu and zenith angle.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_train, load_test
from verify import evaluate


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

    # Extract features
    muon_train = X_train[:, :, :, 1].astype(np.float32)  # (N, 16, 16)
    muon_test = X_test[:, :, :, 1].astype(np.float32)

    # Scalar features: log(Ne) - log(Nmu), zenith angle, log(Nmu)
    scalar_train = np.stack([
        f_train[:, 3] - f_train[:, 4],  # log(Ne) - log(Nmu)
        f_train[:, 1] / 30,  # Zenith angle normalized
        f_train[:, 4],  # log(Nmu)
    ], axis=1).astype(np.float32)

    scalar_test = np.stack([
        f_test[:, 3] - f_test[:, 4],
        f_test[:, 1] / 30,
        f_test[:, 4],
    ], axis=1).astype(np.float32)

    # Normalize muon channels
    mu_mean = muon_train.mean()
    mu_std = muon_train.std()
    muon_train = (muon_train - mu_mean) / (mu_std + 1e-6)
    muon_test = (muon_test - mu_mean) / (mu_std + 1e-6)

    # Normalize scalars
    s_mean = scalar_train.mean(axis=0)
    s_std = scalar_train.std(axis=0)
    scalar_train = (scalar_train - s_mean) / (s_std + 1e-6)
    scalar_test = (scalar_test - s_mean) / (s_std + 1e-6)

    # Train/val split
    rng = np.random.default_rng(42)
    n = len(y_train)
    val_mask = rng.random(n) < 0.2
    train_mask = ~val_mask

    train_mu = torch.from_numpy(muon_train[train_mask])
    train_s = torch.from_numpy(scalar_train[train_mask])
    train_y = torch.from_numpy(y_train[train_mask]).long()

    val_mu = torch.from_numpy(muon_train[val_mask])
    val_s = torch.from_numpy(scalar_train[val_mask])
    val_y = torch.from_numpy(y_train[val_mask]).long()

    test_mu = torch.from_numpy(muon_test)
    test_s = torch.from_numpy(scalar_test)
    test_y = torch.from_numpy(y_test).long()

    print(f"  Train: {len(train_y):,}  Val: {len(val_y):,}")

    class MuonCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.mlp = nn.Sequential(
                nn.Linear(32 * 4 * 4 + 3, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 2),
            )

        def forward(self, muon, scalar):
            x = self.conv(muon.unsqueeze(1))  # (N, 1, 16, 16) -> (N, 32, 4, 4)
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.cat([x, scalar], dim=1)
            return self.mlp(x)

    model = MuonCNN().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    class_weights = torch.tensor(
        [len(y_train) / (2 * n_gamma), len(y_train) / (2 * n_hadron)],
        dtype=torch.float32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    print("\nTraining...")
    for epoch in range(30):
        model.train()
        train_loss = 0
        for i in range(0, len(train_y), 4096):
            end = min(i + 4096, len(train_y))
            mu = train_mu[i:end].to(device)
            s = train_s[i:end].to(device)
            y = train_y[i:end].to(device)
            logits = model(mu, s)
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
                mu = val_mu[i:end].to(device)
                s = val_s[i:end].to(device)
                y = val_y[i:end].to(device)
                logits = model(mu, s)
                val_loss += criterion(logits, y).item() * (end - i)
                all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

        val_loss /= len(val_y)
        val_scores = np.concatenate(all_scores)

        print(f"Epoch {epoch+1:2d}/30: train_loss={train_loss/len(train_y):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = val_scores
            torch.save(model.state_dict(), "model_v2.pt")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_scores = []
    with torch.no_grad():
        for i in range(0, len(test_y), 8192):
            end = min(i + 8192, len(test_y))
            mu = test_mu[i:end].to(device)
            s = test_s[i:end].to(device)
            logits = model(mu, s)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)
    np.savez("predictions_v2.npz", gamma_scores=test_scores)

    metric = evaluate(test_scores, "v2: CNN on muon channel + Ne/Nmu ratio + zenith")
    return metric


if __name__ == "__main__":
    main()
