"""v5: Residual MLP with advanced engineering + log1p preprocessing.

Deeper, wider network with residual connections.
Log-normalize rare features to balance importance.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_train, load_test
from verify import evaluate


def engineer_features(matrices, features):
    """Advanced feature engineering with log1p normalization."""
    N = len(matrices)
    eng = np.zeros((N, 20), dtype=np.float32)

    # Base features (log-normalized)
    eng[:, 0] = features[:, 0]  # E
    eng[:, 1] = features[:, 1] / 30  # Ze normalized
    eng[:, 2] = features[:, 3] - features[:, 4]  # log(Ne) - log(Nmu)

    # Channel statistics
    e = matrices[:, :, :, 0]
    m = matrices[:, :, :, 1]

    e_sum = e.sum(axis=(1, 2))
    m_sum = m.sum(axis=(1, 2))
    e_max = e.max(axis=(1, 2))
    m_max = m.max(axis=(1, 2))

    eng[:, 3] = np.log1p(e_sum)
    eng[:, 4] = np.log1p(m_sum)
    eng[:, 5] = np.log1p(e_max)
    eng[:, 6] = np.log1p(m_max)

    # Sparsity
    e_count = (e > 0).sum(axis=(1, 2))
    m_count = (m > 0).sum(axis=(1, 2))
    eng[:, 7] = e_count / 256
    eng[:, 8] = m_count / 256

    # Ratios
    eng[:, 9] = np.log1p(e_sum / (m_sum + 1e-6))
    eng[:, 10] = np.log1p(m_count / (e_count + 1e-6))

    # Variance
    eng[:, 11] = np.log1p(e.var(axis=(1, 2)))
    eng[:, 12] = np.log1p(m.var(axis=(1, 2)))

    # Flattened stats
    total = matrices[:, :, :, :].sum(axis=(1, 2, 3))
    active = (matrices[:, :, :, :] > 0).sum(axis=(1, 2, 3))
    eng[:, 13] = np.log1p(total)
    eng[:, 14] = active / 512

    # Physical ratios
    eng[:, 15] = np.log1p(features[:, 3])  # log(1 + log10(Ne))
    eng[:, 16] = np.log1p(features[:, 4])  # log(1 + log10(Nmu))
    eng[:, 17] = features[:, 2] / 360  # Azimuth normalized
    eng[:, 18] = np.log1p(e_max / (m_max + 1e-6))
    eng[:, 19] = np.tanh(eng[:, 2])  # Squashed Ne/Nmu ratio

    return eng


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 2)

        # Residual projection for dim mismatch
        self.res_proj = nn.Linear(256, 256)

    def forward(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))
        h = torch.dropout(h, p=0.2, train=self.training)

        # Residual block 1
        res = h
        h = torch.relu(self.bn2(self.fc2(h)))
        h = torch.dropout(h, p=0.2, train=self.training)
        h = self.fc3(h)
        h = torch.relu(self.bn3(h + res))
        h = torch.dropout(h, p=0.15, train=self.training)

        h = torch.relu(self.bn4(self.fc4(h)))
        h = torch.dropout(h, p=0.1, train=self.training)
        return self.fc5(h)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

    print("Engineering features...")
    feat_train = engineer_features(X_train, f_train)
    feat_test = engineer_features(X_test, f_test)

    # Normalize
    mean = feat_train.mean(axis=0)
    std = feat_train.std(axis=0)
    std[std < 1e-6] = 1.0
    feat_train = (feat_train - mean) / std
    feat_test = (feat_test - mean) / std

    # Train/val split
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

    print(f"  Train: {len(train_y):,}  Val: {len(val_y):,}")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4096, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=8192, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=8192, shuffle=False)

    model = ResidualMLP().to(device)
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

        print(f"Epoch {epoch+1:2d}/35: train_loss={train_loss/len(train_y):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = val_scores
            torch.save(model.state_dict(), "model_v5.pt")

    # Evaluate on test
    print("\nEvaluating on test set...")
    model.eval()
    test_scores = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)
    np.savez("predictions_v5.npz", gamma_scores=test_scores)

    metric = evaluate(test_scores, "v5: Residual MLP with log1p features + advanced engineering")
    return metric


if __name__ == "__main__":
    main()
