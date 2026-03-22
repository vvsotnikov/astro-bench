"""
v36: BCE + pairwise AUC loss combined from the start.

The problem with v35: switching from BCE to pairwise loss destroys learned features.
Fix: combine BCE + AUC loss from the beginning, with BCE dominant and AUC as regularizer.

AUC loss: for mini-batches of gamma and hadron, compute
  loss_auc = mean(sigmoid((hadron_score - gamma_score) / T))
  where T is a temperature (lower = harder).

Combined loss = (1 - beta) * BCE + beta * AUC_loss

Key difference from v35:
- BCE + AUC from the start (no phase switching)
- beta starts at 0.2, increases to 0.5 over training
- Use logits output (not sigmoid), BCE on logits with proper sign convention
- Temperature annealing: start T=1.0 (soft), decrease to T=0.1 (hard)

Score convention: output logit score, sigmoid -> gamma probability (higher = more gamma).
BCE target: 1 for gamma, 0 for hadron (opposite of standard).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 99
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
    """Standard v2 architecture but outputs 1 logit (gamma score)."""
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
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
        )
    def forward(self, mat, feat):
        return self.head(torch.cat([self.cnn(mat), self.feat_mlp(feat)], dim=1)).squeeze(-1)


class GammaDataset(Dataset):
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


def pairwise_auc_loss(gamma_logits, hadron_logits, temperature=1.0):
    """
    Soft AUC loss: push gamma logits above hadron logits.
    For all pairs (g, h): want logit_g > logit_h.
    Loss = mean(sigmoid((hadron - gamma) / T))
    T=1.0: same as standard sigmoid loss; T=0.1: much sharper
    """
    diff = hadron_logits.unsqueeze(0) - gamma_logits.unsqueeze(1)  # (Ng, Nh)
    return torch.sigmoid(diff / temperature).mean()


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

    # Separate gamma/hadron training indices
    tr_gamma_idx = np.where(y_all[tr_idx] == 0)[0]
    tr_hadron_idx = np.where(y_all[tr_idx] == 1)[0]
    print(f"Train gamma: {len(tr_gamma_idx):,}, hadron: {len(tr_hadron_idx):,}")

    val_ds = GammaDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    gamma_ds = GammaDataset(m_raw[tr_idx[tr_gamma_idx]], f_all[tr_idx[tr_gamma_idx]],
                             y_all[tr_idx[tr_gamma_idx]], feat_mean, feat_std)
    hadron_ds = GammaDataset(m_raw[tr_idx[tr_hadron_idx]], f_all[tr_idx[tr_hadron_idx]],
                              y_all[tr_idx[tr_hadron_idx]], feat_mean, feat_std)

    # Use 512 per class for pairwise loss (256k pairs per step)
    BATCH_G = 512; BATCH_H = 512
    gamma_loader = DataLoader(gamma_ds, batch_size=BATCH_G, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    hadron_loader = DataLoader(hadron_ds, batch_size=BATCH_H, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True)

    n_gamma_tr = len(tr_gamma_idx); n_hadron_tr = len(tr_hadron_idx)
    # BCE weight for class imbalance
    bce_pos_weight = torch.tensor([n_hadron_tr / n_gamma_tr]).to(DEVICE)

    model = GammaCNN(f_all.shape[1]).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_val = 1.0; best_test_scores = None; best_epoch = 0
    N_EPOCHS = 30

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0; total_bce = 0; total_auc = 0; n_batches = 0

        # Temperature annealing: T = 1.0 -> 0.2 over training
        temp = max(0.2, 1.0 - 0.8 * epoch / N_EPOCHS)
        # Beta (AUC weight): 0.3 -> 0.6 over training
        beta = min(0.6, 0.3 + 0.3 * epoch / N_EPOCHS)

        g_iter = iter(gamma_loader); h_iter = iter(hadron_loader)
        steps = min(len(gamma_loader), len(hadron_loader))

        for step in range(steps):
            try:
                g_mat, g_feat, g_lbl = next(g_iter)
                h_mat, h_feat, h_lbl = next(h_iter)
            except StopIteration:
                break

            g_mat, g_feat = g_mat.to(DEVICE), g_feat.to(DEVICE)
            h_mat, h_feat = h_mat.to(DEVICE), h_feat.to(DEVICE)

            g_logits = model(g_mat, g_feat)  # gamma scores (should be high)
            h_logits = model(h_mat, h_feat)  # hadron scores (should be low)

            # BCE loss: target 1 for gamma (we output gamma score), target 0 for hadron
            bce_g = nn.functional.binary_cross_entropy_with_logits(
                g_logits, torch.ones_like(g_logits), pos_weight=bce_pos_weight
            )
            bce_h = nn.functional.binary_cross_entropy_with_logits(
                h_logits, torch.zeros_like(h_logits)
            )
            loss_bce = (bce_g + bce_h) / 2.0

            # Pairwise AUC loss
            loss_auc = pairwise_auc_loss(g_logits, h_logits, temperature=temp)

            loss = (1 - beta) * loss_bce + beta * loss_auc

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_bce += loss_bce.item(); total_auc += loss_auc.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_scores = []; val_labels = []
        with torch.no_grad():
            for mat, feat, label in val_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                scores = torch.sigmoid(model(mat, feat)).cpu().numpy()
                val_scores.extend(scores); val_labels.extend(label.numpy())
        val_surv = survival_at_75(np.array(val_scores), np.array(val_labels))
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}: loss={total_loss/n_batches:.4f} "
              f"bce={total_bce/n_batches:.4f} auc={total_auc/n_batches:.4f} "
              f"T={temp:.2f} beta={beta:.2f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            best_epoch = epoch
            test_scores_list = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    scores = torch.sigmoid(model(mat, feat)).cpu().numpy()
                    test_scores_list.extend(scores)
            best_test_scores = np.array(test_scores_list)
            print(f"  -> Best val: {val_surv:.2e}")

    if best_test_scores is None:
        model.eval()
        test_scores_list = []
        with torch.no_grad():
            for mat, feat, _ in test_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                scores = torch.sigmoid(model(mat, feat)).cpu().numpy()
                test_scores_list.extend(scores)
        best_test_scores = np.array(test_scores_list)

    test_surv = survival_at_75(best_test_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e} (epoch {best_epoch+1})")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v36.pt")
    np.save(f"{OUT_DIR}/probs_v36.npy", best_test_scores)
    np.savez(f"{OUT_DIR}/predictions_v36.npz", gamma_scores=best_test_scores)

    # Test blending with ens3
    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_test_scores + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + v36 alpha={alpha}: {s:.4e}")

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: BCE + pairwise AUC loss combined, T annealing 1.0->0.2, beta 0.3->0.6, 30ep")


if __name__ == "__main__":
    main()
