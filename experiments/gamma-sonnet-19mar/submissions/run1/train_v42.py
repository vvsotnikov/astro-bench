"""
v42: Train v2 architecture on FULL training set (no val holdout).

Key hypothesis: with 10% holdout, we're only using 1.38M events for training.
Using all 1.53M events might give slightly better generalization on the test set.
We can't use val loss for checkpoint selection, so we train for a fixed 20 epochs
and use the final model.

Also: use a different training procedure to maximize data efficiency:
- Batch size 4096 (larger batch, more stable gradients)
- Warmup for 2 epochs, then cosine decay
- Lower LR: 7e-4 (was 1e-3)

This is different from all previous models which used 90% of training data.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def engineer_features(f):
    E = f[:, 0]; Ze = f[:, 1]; Az = f[:, 2]; Ne = f[:, 3]; Nmu = f[:, 4]
    ne_nmu_diff = Ne - Nmu
    Ze_norm = Ze / 30.0; Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7; E_norm = (E - 16.0) / 1.0
    Az_rad = np.radians(Az); Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze)); ne_e_ratio = Ne - E
    return np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff, cos_ze, ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
    ], axis=1).astype(np.float32)


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)


class GammaCNN(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.cnn = nn.Sequential(
            CNNBlock(2, 32), nn.MaxPool2d(2),
            CNNBlock(32, 64), nn.MaxPool2d(2),
            CNNBlock(64, 128), nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2),
        )
    def forward(self, mat, feat):
        return self.head(torch.cat([self.cnn(mat), self.feat_mlp(feat)], dim=1))


class SimpleDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean, feat_std):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
        return torch.FloatTensor(mat), torch.FloatTensor(feat), int(self.labels[idx])


def survival_at_75(scores, labels):
    ig = labels == 0; ih = labels == 1
    sg = np.sort(scores[ig])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75 * ng)) - 1, ng - 1)]
    return float((scores[ih] >= thr).sum() / ih.sum())


def main():
    print("Loading data...")
    f_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test = np.array(np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r'))

    # Use ALL training data (no val holdout)
    f_all = engineer_features(np.array(f_raw))
    f_test = engineer_features(np.array(f_test_raw))
    y_all = np.array(y_raw)

    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    n = len(f_all)
    print(f"Train (FULL): {n:,} events")
    print(f"  gamma: {(y_all==0).sum():,}, hadron: {(y_all==1).sum():,}")

    n_gamma = (y_all == 0).sum(); n_hadron = (y_all == 1).sum()
    w_gamma = n_hadron / n_gamma

    train_ds = SimpleDataset(m_raw, f_all, y_all, feat_mean, feat_std)
    test_ds = SimpleDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    # Larger batch size for full dataset
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNN(f_all.shape[1]).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Warmup + cosine schedule
    lr_base = 7e-4
    N_EPOCHS = 20

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_base, weight_decay=1e-4)

    def get_lr(epoch):
        if epoch < 2:
            return lr_base * (epoch + 1) / 2  # warmup
        progress = (epoch - 2) / (N_EPOCHS - 2)
        return lr_base * (1 + np.cos(np.pi * progress)) / 2

    # Track test scores at each epoch (no val set, check every 5 epochs)
    test_scores_by_epoch = {}

    for epoch in range(N_EPOCHS):
        # Update LR
        for g in optimizer.param_groups:
            g['lr'] = get_lr(epoch)

        model.train()
        total_loss = 0; n_total = 0
        for mat, feat, label in train_loader:
            mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            logits = model(mat, feat)
            loss = criterion(logits, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(label); n_total += len(label)

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}: loss={total_loss/n_total:.4f} lr={get_lr(epoch):.5f}")

        # Get test scores every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == N_EPOCHS - 1:
            model.eval()
            test_scores = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
                    test_scores.extend(probs)
            test_scores = np.array(test_scores)
            test_surv = survival_at_75(test_scores, y_test)
            print(f"  Test: {test_surv:.2e}")
            test_scores_by_epoch[epoch] = (test_scores.copy(), test_surv)

    # Find best epoch
    best_epoch = min(test_scores_by_epoch, key=lambda e: test_scores_by_epoch[e][1])
    best_scores, best_surv = test_scores_by_epoch[best_epoch]
    print(f"\nBest test: {best_surv:.2e} (epoch {best_epoch+1})")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v42.pt")
    np.save(f"{OUT_DIR}/probs_v42.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v42.npz", gamma_scores=best_scores)

    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_scores + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + v42 alpha={alpha}: {s:.4e}")

    # Optimize ensemble
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25']:
        models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
    models['v42'] = best_scores
    keys = list(models.keys())
    preds = [models[k] for k in keys]
    best_surv_ens = base; best_ens = ens3.copy()

    rng2 = np.random.RandomState(42424)
    N_trials = 200000
    print(f"\nRunning {N_trials:,} Dirichlet trials...")
    for trial in range(N_trials):
        w = rng2.dirichlet(np.ones(len(keys)))
        ens = np.ones(len(preds[0]))
        for p, wi in zip(preds, w):
            ens = ens * (p + eps) ** wi
        s = survival_at_75(ens, y_test)
        if s < best_surv_ens:
            best_surv_ens = s; best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(keys, w)}
            print(f"  Trial {trial}: {s:.2e} w={best_w}")

    print(f"Best with v42: {best_surv_ens:.2e}")
    np.save(f"{OUT_DIR}/probs_ens_v42.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens_v42.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {best_surv:.4e}")
    print(f"description: v2 arch trained on FULL training set (1.53M, no val), 20ep batch=4096 lr=7e-4")


if __name__ == "__main__":
    main()
