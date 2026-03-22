"""v6: RF + NN Hybrid ensemble.

Stacking approach: Random Forest + Neural Network ensemble.
Stack two feature engineering approaches for complementary strengths.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier

from load_data import load_train, load_test
from verify import evaluate


def features_set1(matrices, features):
    """Set 1: Spatial + statistics focus."""
    N = len(matrices)
    eng = np.zeros((N, 12), dtype=np.float32)
    e = matrices[:, :, :, 0]
    m = matrices[:, :, :, 1]
    eng[:, 0] = features[:, 0]  # E
    eng[:, 1] = features[:, 3] - features[:, 4]  # Ne - Nmu
    eng[:, 2] = np.log1p(e.sum(axis=(1, 2)))
    eng[:, 3] = np.log1p(m.sum(axis=(1, 2)))
    eng[:, 4] = np.log1p(e.max(axis=(1, 2)))
    eng[:, 5] = np.log1p(m.max(axis=(1, 2)))
    eng[:, 6] = (e > 0).sum(axis=(1, 2)) / 256
    eng[:, 7] = (m > 0).sum(axis=(1, 2)) / 256
    eng[:, 8] = e.var(axis=(1, 2))
    eng[:, 9] = m.var(axis=(1, 2))
    eng[:, 10] = features[:, 1] / 30  # Zenith
    eng[:, 11] = np.log1p(features[:, 3])  # log(Ne)
    return eng


def features_set2(matrices, features):
    """Set 2: Ratio & physics focus."""
    N = len(matrices)
    eng = np.zeros((N, 10), dtype=np.float32)
    e = matrices[:, :, :, 0]
    m = matrices[:, :, :, 1]
    e_sum = e.sum(axis=(1, 2))
    m_sum = m.sum(axis=(1, 2))
    eng[:, 0] = features[:, 3] - features[:, 4]  # Ne - Nmu
    eng[:, 1] = np.log1p(e_sum / (m_sum + 1e-6))
    eng[:, 2] = np.log1p((e > 0).sum(axis=(1, 2)) / ((m > 0).sum(axis=(1, 2)) + 1e-6))
    eng[:, 3] = features[:, 0] / 18  # Energy norm
    eng[:, 4] = features[:, 1] / 30  # Zenith norm
    eng[:, 5] = np.log1p(features[:, 4])  # log(Nmu)
    eng[:, 6] = np.log1p(e_sum)
    eng[:, 7] = np.log1p(m_sum)
    eng[:, 8] = (matrices[:, :, :, :] > 0).sum(axis=(1, 2, 3)) / 512
    eng[:, 9] = np.tanh(features[:, 3] - features[:, 4])
    return eng


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

    # Feature sets
    print("Engineering features...")
    f1_train = features_set1(X_train, f_train)
    f1_test = features_set1(X_test, f_test)
    f2_train = features_set2(X_train, f_train)
    f2_test = features_set2(X_test, f_test)

    # Normalize both
    for f in [f1_train, f2_train, f1_test, f2_test]:
        m = f.mean(axis=0)
        s = f.std(axis=0)
        s[s < 1e-6] = 1.0
        f[:] = (f - m) / s

    # Train RF on set1
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=80, max_depth=18, n_jobs=8, random_state=42)
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    w = np.where(y_train == 0, len(y_train) / (2 * n_gamma), len(y_train) / (2 * n_hadron))
    rf.fit(f1_train, y_train, sample_weight=w)
    rf_proba = rf.predict_proba(f1_test)[:, 0]

    # Train NN on set2
    print("Training Neural Network...")
    rng = np.random.default_rng(42)
    n = len(y_train)
    val_mask = rng.random(n) < 0.2
    train_mask = ~val_mask

    train_x = torch.from_numpy(f2_train[train_mask]).float()
    train_y = torch.from_numpy(y_train[train_mask]).long()
    test_x = torch.from_numpy(f2_test).float()

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4096, shuffle=True)

    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.net(x)

    model = SmallMLP().to(device)
    class_weights = torch.tensor(
        [len(y_train) / (2 * n_gamma), len(y_train) / (2 * n_hadron)],
        dtype=torch.float32
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Training NN...")
    for epoch in range(25):
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
    nn_proba = []
    with torch.no_grad():
        for i in range(0, len(test_x), 8192):
            x = test_x[i:i+8192].to(device)
            logits = model(x)
            nn_proba.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())
    nn_proba = np.concatenate(nn_proba)

    # Ensemble: simple average
    ensemble_scores = 0.5 * rf_proba + 0.5 * nn_proba
    np.savez("predictions_v6.npz", gamma_scores=ensemble_scores)

    metric = evaluate(ensemble_scores, "v6: RF (set1) + NN (set2) ensemble (50/50 average)")
    return metric


if __name__ == "__main__":
    main()
