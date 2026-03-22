"""v8: Deeper MLP with aggressive regularization.

Try deeper network with careful dropout to avoid overfitting.
More epochs for better convergence.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_train, load_test
from verify import evaluate


def engineer_features(matrices, features):
    """Enhanced feature engineering (24 features)."""
    N = len(matrices)
    eng = np.zeros((N, 24), dtype=np.float32)

    eng[:, :5] = features[:, :5]

    e = matrices[:, :, :, 0]
    m = matrices[:, :, :, 1]

    eng[:, 5] = np.log1p(m.sum(axis=(1, 2)))
    eng[:, 6] = np.log1p(m.max(axis=(1, 2)))
    eng[:, 7] = (m > 0).sum(axis=(1, 2)) / 256
    eng[:, 8] = np.log1p(e.sum(axis=(1, 2)))
    eng[:, 9] = np.log1p(e.max(axis=(1, 2)))
    eng[:, 10] = (e > 0).sum(axis=(1, 2)) / 256

    ne = features[:, 3]
    nmu = features[:, 4]
    eng[:, 11] = ne - nmu
    eng[:, 12] = np.tanh(ne - nmu)  # Squashed version

    eng[:, 13] = (m > 0).sum(axis=(1, 2)) / ((e > 0).sum(axis=(1, 2)) + 1e-6)
    eng[:, 14] = np.log1p(m.sum(axis=(1, 2)) / (e.sum(axis=(1, 2)) + 1e-6))
    eng[:, 15] = np.log1p(m.var(axis=(1, 2)))
    eng[:, 16] = np.log1p(e.var(axis=(1, 2)))

    for i in range(N):
        e_i = e[i]
        m_i = m[i]
        if e_i.sum() > 0:
            cy, cx = np.indices(e_i.shape)
            eng[i, 17] = np.average(cy, weights=e_i)
            eng[i, 18] = np.average(cx, weights=e_i)
        if m_i.sum() > 0:
            cy, cx = np.indices(m_i.shape)
            eng[i, 19] = np.average(cy, weights=m_i)
            eng[i, 20] = np.average(cx, weights=m_i)

    total = matrices[:, :, :, :].sum(axis=(1, 2, 3))
    active = (matrices[:, :, :, :] > 0).sum(axis=(1, 2, 3))
    eng[:, 21] = np.log1p(total)
    eng[:, 22] = active / 512
    eng[:, 23] = np.log1p(total / (active + 1))

    return eng


class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 384),
            nn.BatchNorm1d(384),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(384, 384),
            nn.BatchNorm1d(384),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


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

    model = DeepMLP().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    class_weights = torch.tensor(
        [len(y_train) / (2 * n_gamma), len(y_train) / (2 * n_hadron)],
        dtype=torch.float32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    print("\nTraining...")
    for epoch in range(50):
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

        if epoch % 10 == 0 or epoch == 49:
            print(f"Epoch {epoch+1:2d}/50: train_loss={train_loss/len(train_y):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = val_scores
            torch.save(model.state_dict(), "model_v8.pt")

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
    np.savez("predictions_v8.npz", gamma_scores=test_scores)

    metric = evaluate(test_scores, "v8: Deep MLP (5 layers, 384-256-128-64) with aggressive dropout")
    return metric


if __name__ == "__main__":
    main()
