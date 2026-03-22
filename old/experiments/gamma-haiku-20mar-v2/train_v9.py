"""v9: Simple ensemble of v1 (MLP) and v4 (RF).

Best so far are v1 (7.30e-04) and v4 (2.04e-03).
Combining them might exploit complementary strengths.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier

from load_data import load_train, load_test
from verify import evaluate


def engineer_features_v1(matrices, features):
    """v1 features (23)."""
    N = len(matrices)
    eng = np.zeros((N, 23), dtype=np.float32)
    eng[:, :5] = features[:, :5]
    e = matrices[:, :, :, 0]
    m = matrices[:, :, :, 1]
    eng[:, 5] = m.sum(axis=(1, 2))
    eng[:, 6] = m.max(axis=(1, 2))
    eng[:, 7] = (m > 0).sum(axis=(1, 2))
    eng[:, 8] = e.sum(axis=(1, 2))
    eng[:, 9] = e.max(axis=(1, 2))
    eng[:, 10] = (e > 0).sum(axis=(1, 2))
    ne = features[:, 3]
    nmu = features[:, 4]
    eng[:, 11] = ne - nmu
    eng[:, 12] = (m > 0).sum(axis=(1, 2)) / ((e > 0).sum(axis=(1, 2)) + 1e-6)
    eng[:, 13] = m.sum(axis=(1, 2)) / (e.sum(axis=(1, 2)) + 1e-6)
    eng[:, 14] = m.var(axis=(1, 2))
    eng[:, 15] = e.var(axis=(1, 2))
    for i in range(N):
        e_i = e[i]
        m_i = m[i]
        if e_i.sum() > 0:
            cy, cx = np.indices(e_i.shape)
            eng[i, 16] = np.average(cy, weights=e_i)
            eng[i, 17] = np.average(cx, weights=e_i)
        if m_i.sum() > 0:
            cy, cx = np.indices(m_i.shape)
            eng[i, 18] = np.average(cy, weights=m_i)
            eng[i, 19] = np.average(cx, weights=m_i)
    eng[:, 20] = matrices[:, :, :, :].sum(axis=(1, 2, 3))
    eng[:, 21] = (matrices[:, :, :, :] > 0).sum(axis=(1, 2, 3))
    eng[:, 22] = np.log1p(eng[:, 20] / (eng[:, 21] + 1))
    return eng


def engineer_features_v4(matrices, features):
    """v4 features (15)."""
    N = len(matrices)
    eng = np.zeros((N, 15), dtype=np.float32)
    eng[:, 0] = features[:, 0]
    eng[:, 1] = features[:, 1]
    eng[:, 2] = features[:, 3] - features[:, 4]
    e = matrices[:, :, :, 0]
    m = matrices[:, :, :, 1]
    eng[:, 3] = e.sum(axis=(1, 2))
    eng[:, 4] = m.sum(axis=(1, 2))
    eng[:, 5] = e.max(axis=(1, 2))
    eng[:, 6] = m.max(axis=(1, 2))
    eng[:, 7] = (e > 0).sum(axis=(1, 2))
    eng[:, 8] = (m > 0).sum(axis=(1, 2))
    eng[:, 9] = np.log1p(e.sum(axis=(1, 2)) / (m.sum(axis=(1, 2)) + 1e-6))
    eng[:, 10] = (m > 0).sum(axis=(1, 2)) / ((e > 0).sum(axis=(1, 2)) + 1e-6)
    eng[:, 11] = e.var(axis=(1, 2))
    eng[:, 12] = m.var(axis=(1, 2))
    eng[:, 13] = (matrices[:, :, :, :] > 0).sum(axis=(1, 2, 3))
    eng[:, 14] = matrices[:, :, :, :].sum(axis=(1, 2, 3))
    return eng


class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
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

    # Feature set 1 (v1)
    print("Engineering features...")
    feat1_train = engineer_features_v1(X_train, f_train)
    feat1_test = engineer_features_v1(X_test, f_test)
    m1 = feat1_train.mean(axis=0)
    s1 = feat1_train.std(axis=0)
    s1[s1 < 1e-6] = 1.0
    feat1_train = (feat1_train - m1) / s1
    feat1_test = (feat1_test - m1) / s1

    # Feature set 2 (v4)
    feat2_train = engineer_features_v4(X_train, f_train)
    feat2_test = engineer_features_v4(X_test, f_test)
    m2 = feat2_train.mean(axis=0)
    s2 = feat2_train.std(axis=0)
    s2[s2 < 1e-6] = 1.0
    feat2_train = (feat2_train - m2) / s2
    feat2_test = (feat2_test - m2) / s2

    # Train MLP on feature set 1
    print("Training MLP...")
    rng = np.random.default_rng(42)
    n = len(y_train)
    val_mask = rng.random(n) < 0.2
    train_mask = ~val_mask

    train_x1 = torch.from_numpy(feat1_train[train_mask]).float()
    train_y = torch.from_numpy(y_train[train_mask]).long()
    val_x1 = torch.from_numpy(feat1_train[val_mask]).float()
    val_y = torch.from_numpy(y_train[val_mask]).long()
    test_x1 = torch.from_numpy(feat1_test).float()

    train_loader = DataLoader(TensorDataset(train_x1, train_y), batch_size=4096, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x1, val_y), batch_size=8192, shuffle=False)

    model = SimpleMLP(23).to(device)
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

    for epoch in range(30):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = np.concatenate(all_scores)

    model.eval()
    test_scores_mlp = []
    with torch.no_grad():
        for i in range(0, len(test_x1), 8192):
            x = test_x1[i:i+8192].to(device)
            logits = model(x)
            test_scores_mlp.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())
    test_scores_mlp = np.concatenate(test_scores_mlp)

    # Train RF on feature set 2
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=8, random_state=42)
    w = np.where(y_train == 0, len(y_train) / (2 * n_gamma), len(y_train) / (2 * n_hadron))
    rf.fit(feat2_train, y_train, sample_weight=w)
    test_scores_rf = rf.predict_proba(feat2_test)[:, 0]

    # Ensemble: average
    ensemble_scores = 0.5 * test_scores_mlp + 0.5 * test_scores_rf
    np.savez("predictions_v9.npz", gamma_scores=ensemble_scores)

    metric = evaluate(ensemble_scores, "v9: Ensemble MLP (v1 features) + RF (v4 features), 50/50")
    return metric


if __name__ == "__main__":
    main()
