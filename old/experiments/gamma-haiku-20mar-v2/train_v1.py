"""v1: MLP with physics-informed feature engineering.

Key insight: Gamma rays produce no muons (Nmu ≈ 3.0 vs 3.5 for hadrons).
Use engineered features: muon channel energy, Ne/Nmu ratio, spatial moments.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_train, load_test
from verify import evaluate


def engineer_features(matrices, features):
    """Create physics-informed features.

    Args:
        matrices: (N, 16, 16, 2) - channel 0: electrons, channel 1: muons
        features: (N, 5) - E, Ze, Az, Ne, Nmu

    Returns:
        (N, 23) feature matrix
    """
    N = len(matrices)
    eng = np.zeros((N, 23), dtype=np.float32)

    # Original 5 features
    eng[:, :5] = features[:, :5]

    # Channel separation
    electron_channel = matrices[:, :, :, 0]  # (N, 16, 16)
    muon_channel = matrices[:, :, :, 1]      # (N, 16, 16)

    # Muon statistics (indices 5-10)
    eng[:, 5] = muon_channel.sum(axis=(1, 2))  # Total muon energy
    eng[:, 6] = muon_channel.max(axis=(1, 2))  # Max muon
    eng[:, 7] = (muon_channel > 0).sum(axis=(1, 2))  # Muon cell count
    eng[:, 8] = electron_channel.sum(axis=(1, 2))  # Total electron energy
    eng[:, 9] = electron_channel.max(axis=(1, 2))  # Max electron
    eng[:, 10] = (electron_channel > 0).sum(axis=(1, 2))  # Electron cell count

    # Ratios and combinations (indices 11-15)
    # Ne/Nmu ratio - direct physics discriminator
    ne = features[:, 3]  # log10(Ne)
    nmu = features[:, 4]  # log10(Nmu)
    eng[:, 11] = ne - nmu  # log10(Ne) - log10(Nmu) = log10(Ne/Nmu)

    eng[:, 12] = (muon_channel > 0).sum(axis=(1, 2)) / (electron_channel > 0).sum(axis=(1, 2)) + 1e-6
    eng[:, 13] = muon_channel.sum(axis=(1, 2)) / (electron_channel.sum(axis=(1, 2)) + 1e-6)
    eng[:, 14] = muon_channel.var(axis=(1, 2))  # Muon distribution variance
    eng[:, 15] = electron_channel.var(axis=(1, 2))  # Electron distribution variance

    # Spatial moments (indices 16-22)
    for i in range(N):
        e = electron_channel[i]
        m = muon_channel[i]
        if e.sum() > 0:
            cy, cx = np.indices(e.shape)
            eng[i, 16] = np.average(cy, weights=e)  # Electron centroid Y
            eng[i, 17] = np.average(cx, weights=e)  # Electron centroid X
        if m.sum() > 0:
            cy, cx = np.indices(m.shape)
            eng[i, 18] = np.average(cy, weights=m)  # Muon centroid Y
            eng[i, 19] = np.average(cx, weights=m)  # Muon centroid X

    # Flatten matrix statistics
    eng[:, 20] = matrices[:, :, :, :].sum(axis=(1, 2, 3))  # Total detector energy
    eng[:, 21] = (matrices[:, :, :, :] > 0).sum(axis=(1, 2, 3))  # Active cells
    eng[:, 22] = np.log1p(eng[:, 20] / (eng[:, 21] + 1))  # Log energy per active cell

    return eng


def main():
    device = torch.device("cuda:0")  # cuda:0 when CUDA_VISIBLE_DEVICES=1
    print(f"Device: {device}")

    # Load and engineer features
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

    print("Engineering features...")
    feat_train = engineer_features(X_train, f_train)
    feat_test = engineer_features(X_test, f_test)
    print(f"  Train features: {feat_train.shape}")

    # Normalize
    mean = feat_train.mean(axis=0)
    std = feat_train.std(axis=0)
    std[std < 1e-6] = 1.0
    feat_train = (feat_train - mean) / std
    feat_test = (feat_test - mean) / std

    # Train/val split (80/20 on training data)
    rng = np.random.default_rng(42)
    n = len(y_train)
    val_mask = rng.random(n) < 0.2
    train_mask = ~val_mask

    train_x = torch.from_numpy(feat_train[train_mask]).float()
    train_y = torch.from_numpy(y_train[train_mask]).long()
    val_x = torch.from_numpy(feat_train[val_mask]).float()
    val_y = torch.from_numpy(y_train[val_mask]).long()
    test_x = torch.from_numpy(feat_test).float()
    test_y = torch.from_numpy(y_test).long()

    print(f"  Train split: {len(train_y):,}  Val: {len(val_y):,}")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4096, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=8192, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=8192, shuffle=False)

    # Model
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(23, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    print("\nTraining...")
    for epoch in range(40):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        all_scores = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

        val_loss /= len(val_y)
        val_scores = np.concatenate(all_scores)

        print(f"Epoch {epoch+1:2d}/40: train_loss={train_loss/len(train_y):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = val_scores
            torch.save(model.state_dict(), "model_v1.pt")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_scores = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)
    np.savez("predictions_v1.npz", gamma_scores=test_scores)

    metric = evaluate(test_scores, "v1: MLP with physics features (Ne/Nmu, muon stats, spatial moments)")
    return metric


if __name__ == "__main__":
    main()
