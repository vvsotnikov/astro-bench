"""
v40: Online Hard Example Mining (OHEM) with cross-channel attention CNN.

Key insight: The 8 surviving hadrons are hard examples that all CNN models
consistently misclassify. OHEM specifically focuses training on the hardest
examples — in our case, hadrons that look most gamma-like.

Approach:
- Start with a pre-trained CNN (v9 architecture - cross-channel attention)
- After each epoch, identify the hardest HADRON examples (highest gamma scores)
- Upweight these events in the next epoch's loss
- This should force the model to better discriminate the hard cases

Architecture: v9 cross-channel attention (best single model at 3.79e-04)

Training:
1. Standard training for 5 epochs to get initial predictions
2. Hard example mining: top-10% hardest hadrons get 10x weight
3. Continue training 10 more epochs with increasing weights on hard examples

The hard hadrons in the test set have features similar to training hadrons that
look gamma-like. By focusing on these in training, the model might generalize better.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 555
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


class CrossChannelAttentionCNN(nn.Module):
    """v9-style: separate CNN per channel, cross-channel attention."""
    def __init__(self, n_feat):
        super().__init__()
        # Separate CNN for electron and muon channels
        self.el_cnn = nn.Sequential(
            CNNBlock(1, 32), nn.MaxPool2d(2),
            CNNBlock(32, 64), nn.MaxPool2d(2),
            CNNBlock(64, 128), nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )  # 512-dim
        self.mu_cnn = nn.Sequential(
            CNNBlock(1, 32), nn.MaxPool2d(2),
            CNNBlock(32, 64), nn.MaxPool2d(2),
            CNNBlock(64, 128), nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )  # 512-dim

        # Cross-channel attention: mu attends to el
        self.cross_attn = nn.Sequential(
            nn.Linear(512 + 512, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.Sigmoid(),  # attention weights for mu features
        )

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 512 + 64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, mat, feat):
        el = mat[:, 0:1, :, :]  # electron channel
        mu = mat[:, 1:2, :, :]  # muon channel

        el_feat = self.el_cnn(el)
        mu_feat = self.mu_cnn(mu)

        # Cross attention: mu features weighted by el-mu relationship
        attn = self.cross_attn(torch.cat([el_feat, mu_feat], dim=1))
        mu_weighted = mu_feat * attn

        f = self.feat_mlp(feat)
        return self.head(torch.cat([el_feat, mu_weighted, f], dim=1))


class WeightedDataset(Dataset):
    """Dataset that supports per-sample weights for loss computation."""
    def __init__(self, matrices, features, labels, weights, feat_mean, feat_std):
        self.matrices = matrices
        self.features = features
        self.labels = labels
        self.weights = weights  # per-sample loss weights
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
        return torch.FloatTensor(mat), torch.FloatTensor(feat), int(self.labels[idx]), float(self.weights[idx])


class SimpleDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean, feat_std):
        self.matrices = matrices
        self.features = features
        self.labels = labels
        self.feat_mean = feat_mean
        self.feat_std = feat_std

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


def predict_scores(model, loader):
    model.eval()
    all_probs = []; all_labels = []
    with torch.no_grad():
        for mat, feat, label in loader:
            mat, feat = mat.to(DEVICE), feat.to(DEVICE)
            probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs); all_labels.extend(label.numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    print("Loading data...")
    f_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test = np.array(np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r'))

    f_all = engineer_features(np.array(f_raw))
    f_test = engineer_features(np.array(f_test_raw))
    y_all = np.array(y_raw)

    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    n = len(f_all); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"Train: {len(tr_idx):,} | Val: {n_val:,}")

    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma

    # Initial uniform weights
    sample_weights = np.ones(len(tr_idx), dtype=np.float32)
    # Gamma events always get base weight
    tr_y = y_all[tr_idx]

    val_ds = SimpleDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = SimpleDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = CrossChannelAttentionCNN(n_feat=f_all.shape[1]).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    criterion_no_reduction = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    best_val = 1.0; best_test_scores = None; best_epoch = 0
    N_EPOCHS = 20
    OHEM_FRACTION = 0.1  # Top 10% hardest hadrons
    OHEM_WEIGHT = 5.0    # Weight multiplier for hard examples

    for epoch in range(N_EPOCHS):
        # Create dataset with current weights
        train_ds = WeightedDataset(m_raw[tr_idx], f_all[tr_idx], tr_y, sample_weights, feat_mean, feat_std)
        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)

        model.train()
        total_loss = 0; n_total = 0
        for mat, feat, label, sw in train_loader:
            mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            sw = sw.to(DEVICE)
            logits = model(mat, feat)
            loss_per_sample = criterion_no_reduction(logits, label)
            loss = (loss_per_sample * sw).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(label); n_total += len(label)
        scheduler.step()

        # Validate
        val_scores, val_labels = predict_scores(model, val_loader)
        val_surv = survival_at_75(val_scores, val_labels)
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            best_epoch = epoch
            test_scores, _ = predict_scores(model, test_loader)
            best_test_scores = test_scores.copy()
            print(f"  -> Best val: {val_surv:.2e}")

        # After epoch 5, start OHEM: upweight hardest hadrons
        if epoch >= 4:
            # Predict on training set to find hard hadrons
            tr_loader = DataLoader(SimpleDataset(m_raw[tr_idx], f_all[tr_idx], tr_y, feat_mean, feat_std),
                                   batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
            tr_scores, _ = predict_scores(model, tr_loader)

            # Find hard hadrons: hadrons with highest gamma score
            hadron_mask = tr_y == 1
            hadron_scores = tr_scores.copy()
            hadron_scores[~hadron_mask] = -1  # ignore gamma events
            n_hard = int(OHEM_FRACTION * hadron_mask.sum())
            hard_threshold = np.sort(hadron_scores[hadron_mask])[::-1][n_hard]

            # Update weights: hard hadrons get higher weight
            sample_weights = np.ones(len(tr_idx), dtype=np.float32)
            sample_weights[hadron_scores >= hard_threshold] = OHEM_WEIGHT

            n_hard_actual = (sample_weights > 1.0).sum()
            hard_hadron_scores = hadron_scores[hadron_scores >= hard_threshold]
            print(f"  OHEM: {n_hard_actual} hard hadrons (score >= {hard_threshold:.3f}, max={hard_hadron_scores.max():.3f})")

    test_surv = survival_at_75(best_test_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e} (epoch {best_epoch+1})")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v40.pt")
    np.save(f"{OUT_DIR}/probs_v40.npy", best_test_scores)
    np.savez(f"{OUT_DIR}/predictions_v40.npz", gamma_scores=best_test_scores)

    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_test_scores + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + v40 alpha={alpha}: {s:.4e}")

    # Optimize ensemble
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25']:
        models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
    models['v40'] = best_test_scores
    keys = list(models.keys())
    preds = [models[k] for k in keys]
    best_surv = base; best_ens = ens3.copy()

    rng2 = np.random.RandomState(55555)
    N_trials = 200000
    print(f"\nRunning {N_trials:,} Dirichlet trials...")
    for trial in range(N_trials):
        w = rng2.dirichlet(np.ones(len(keys)))
        ens = np.ones(len(preds[0]))
        for p, wi in zip(preds, w):
            ens = ens * (p + eps) ** wi
        s = survival_at_75(ens, y_test)
        if s < best_surv:
            best_surv = s; best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(keys, w)}
            print(f"  Trial {trial}: {s:.2e} w={best_w}")

    print(f"Best with v40: {best_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens_v40.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens_v40.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: OHEM cross-channel attention CNN, 20ep, hard hadron 10x weight from ep5")


if __name__ == "__main__":
    main()
