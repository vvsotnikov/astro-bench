"""
v1: Feature-engineered MLP with Ne/Nmu ratio as primary discriminant.
Physics: gamma showers have nearly zero muons -> much higher Ne/Nmu ratio.
Uses 5 scalar features + engineered features. Quick baseline.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def engineer_features(f):
    """Build feature vector from raw 5-feature array [E, Ze, Az, Ne, Nmu]."""
    E = f[:, 0]
    Ze = f[:, 1]
    Az = f[:, 2]
    Ne = f[:, 3]
    Nmu = f[:, 4]

    # Core discriminant: Ne - Nmu (in log space = log ratio)
    ne_nmu_diff = Ne - Nmu  # large for gamma, small for hadron

    # Normalized features
    Ze_norm = Ze / 30.0
    Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7
    E_norm = (E - 16.0) / 1.0

    # Trig encoding for azimuth
    Az_rad = np.radians(Az)
    Az_cos = np.cos(Az_rad)
    Az_sin = np.sin(Az_rad)

    # Physics-motivated features
    cos_ze = np.cos(np.radians(Ze))
    ne_e_ratio = Ne - E  # Ne normalized by energy

    feats = np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff,   # PRIMARY: log(Ne/Nmu)
        cos_ze,
        ne_e_ratio,
        Ne * Ze_norm,  # interaction
        Nmu * cos_ze,  # muon * zenith correction
    ], axis=1).astype(np.float32)
    return feats


def load_data():
    # Train
    f_train = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    y_train = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    # Test
    f_test = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    y_test = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r')

    print(f"Train: {len(f_train):,} (gamma={int((y_train==0).sum()):,}, hadron={int((y_train==1).sum()):,})")
    print(f"Test:  {len(f_test):,} (gamma={int((y_test==0).sum()):,}, hadron={int((y_test==1).sum()):,})")

    X_train = engineer_features(np.array(f_train))
    X_test = engineer_features(np.array(f_test))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Train/val split (90/10 of training)
    n = len(X_train)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    n_val = int(n * 0.1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    return (X_train[tr_idx], y_train[tr_idx],
            X_train[val_idx], y_train[val_idx],
            X_test, y_test)


class GammaMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256),
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
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def survival_at_75(scores, labels):
    """Compute hadronic survival at 75% gamma efficiency."""
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])[::-1]
    ng = len(sg)
    idx = int(np.ceil(0.75 * ng)) - 1
    thr = sg[min(idx, ng - 1)]
    survival = (scores[is_hadron] >= thr).sum() / is_hadron.sum()
    return survival


def main():
    print("Loading data...")
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_data()
    print(f"Feature dim: {X_tr.shape[1]}")

    # Class weights (gamma ~5% in training)
    n_gamma = (y_tr == 0).sum()
    n_hadron = (y_tr == 1).sum()
    w_gamma = (n_hadron / n_gamma)  # heavily upweight gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: gamma={w_gamma:.1f}, hadron=1.0")

    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.LongTensor(y_tr)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)

    model = GammaMLP(X_tr.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_survival = 1.0
    best_scores_test = None
    n_epochs = 20

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n_total += len(y)

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            # Validation survival
            val_logits = model(X_val_t.to(DEVICE))
            val_probs = torch.softmax(val_logits, dim=1)[:, 0].cpu().numpy()
            val_survival = survival_at_75(val_probs, y_val)

            # Test scores for checkpoint
            test_logits = model(X_test_t.to(DEVICE))
            test_probs = torch.softmax(test_logits, dim=1)[:, 0].cpu().numpy()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{n_epochs}: loss={total_loss/n_total:.4f} "
              f"val_survival@75={val_survival:.2e} lr={lr:.5f}")

        if val_survival < best_survival:
            best_survival = val_survival
            best_scores_test = test_probs
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v1.pt")
            print(f"  -> New best val survival: {val_survival:.2e}")

    print(f"\nBest val survival@75: {best_survival:.2e}")
    np.save(f"{OUT_DIR}/probs_v1.npy", best_scores_test)
    np.savez(f"{OUT_DIR}/predictions_v1.npz", gamma_scores=best_scores_test)
    np.savez(f"{OUT_DIR}/predictions.npz", gamma_scores=best_scores_test)
    print(f"Saved predictions ({len(best_scores_test)} scores)")

    print("\n---")
    print(f"metric: {best_survival:.4e}")
    print("description: Feature-engineered MLP, Ne/Nmu ratio, 256-256-128, 20 epochs")


if __name__ == "__main__":
    main()
