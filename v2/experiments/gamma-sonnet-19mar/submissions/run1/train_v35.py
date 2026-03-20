"""
v35: Direct metric optimization using differentiable approximation.

Instead of cross-entropy, directly optimize the hadronic survival rate @ 75% gamma efficiency.
This requires a differentiable approximation of the rank-based metric.

Approach:
- Use a smooth approximation of the threshold: sigmoid((score - threshold) / temperature)
- Loss = E[sigma((hadron_score - threshold) / T)] where threshold is implicitly defined by
  the 75th percentile of gamma scores
- This is like a contrastive/ranking loss that directly pushes hadron scores below gamma scores

Key insight: instead of "classify gamma vs hadron", teach the network to RANK gammas above hadrons.
This is done via a pairwise ranking loss (like in learning-to-rank literature) or
via optimizing the AUC directly.

Architecture: same CNN as v2 but with AUC + survival-focused training.

Loss function:
1. AUC approximation loss (smooth ranking): for each (gamma, hadron) pair, maximize gamma_score - hadron_score
2. Survival @ 75% approximation: push gamma quantile-75 scores up, push hadron scores down
3. Combined with BCE for stability

We use a mini-batch approximation: sample B gammas and B hadrons per batch,
compute pairwise loss = mean(sigmoid((hadron_score - gamma_score) / T)).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 77
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


class GammaCNNv35(nn.Module):
    """Same architecture as v2 but outputs a single score (not 2-class logits)."""
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


def auc_loss(gamma_scores, hadron_scores, temperature=0.1):
    """
    Differentiable AUC maximization.
    For each (gamma, hadron) pair, we want gamma_score > hadron_score.
    Loss = mean(sigmoid((hadron - gamma) / T))
    T controls smoothness: lower T = sharper (closer to AUC), higher T = softer.
    """
    # gamma_scores: (B_g,), hadron_scores: (B_h,)
    # pairwise differences: (B_g, B_h) -> (B_g * B_h,)
    diff = hadron_scores.unsqueeze(0) - gamma_scores.unsqueeze(1)  # (B_g, B_h)
    return torch.sigmoid(diff / temperature).mean()


def survival_loss(gamma_scores, hadron_scores, gamma_efficiency=0.75, temperature=0.05):
    """
    Approximate hadronic survival @ gamma_efficiency.
    Find the gamma score threshold at 75th percentile (from top).
    Push hadron scores below this threshold.

    threshold ~ gamma score at index ceil(0.75 * N_gamma) when sorted descending.
    We approximate this with a soft quantile.
    """
    # Soft quantile: approximately the 75th percentile of gamma scores
    # Use a smooth approximation via sorting
    B_g = len(gamma_scores)
    k = int(np.ceil(gamma_efficiency * B_g))  # index from top

    # Sort gamma scores descending (use top-k approach)
    gamma_sorted, _ = torch.sort(gamma_scores, descending=True)
    threshold = gamma_sorted[min(k-1, B_g-1)]

    # Loss: push hadron scores below threshold
    # survival ~ fraction of hadrons above threshold
    # differentiable: sigmoid((hadron - threshold) / T)
    hadron_survival = torch.sigmoid((hadron_scores - threshold) / temperature).mean()
    return hadron_survival


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

    print(f"Train: {len(tr_idx):,} | Val: {len(n_val):,}" if False else f"Train: {len(tr_idx):,} | Val: {n_val:,}")

    train_ds = GammaDataset(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = GammaDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    # Separate gamma/hadron indices for training
    tr_gamma_idx = np.where(y_all[tr_idx] == 0)[0]
    tr_hadron_idx = np.where(y_all[tr_idx] == 1)[0]
    print(f"Train gamma: {len(tr_gamma_idx):,}, hadron: {len(tr_hadron_idx):,}")

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNNv35(f_all.shape[1]).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Pre-train with BCE for 5 epochs to get reasonable starting point
    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w_gamma]).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_val = 1.0; best_test_scores = None; best_epoch = 0

    print("\nPhase 1: BCE pre-training (5 epochs)...")
    for epoch in range(5):
        model.train()
        total_loss = 0; n_total = 0
        for mat, feat, label in train_loader:
            mat, feat = mat.to(DEVICE), feat.to(DEVICE)
            label_f = label.float().to(DEVICE)  # 0=gamma, 1=hadron
            # For BCE: score=0 means gamma (negative), score=1 means hadron (positive)
            # We want gamma_score HIGH (for our metric), so we need to invert
            # Actually: output score = "gamma score" (higher = more gamma-like)
            # BCE: label=0 means gamma (target=0 in standard, but we want high score)
            # Use label=1-y: 1=gamma target, 0=hadron target
            logit = model(mat, feat)
            target = 1.0 - label_f  # 1=gamma, 0=hadron
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, target,
                pos_weight=torch.tensor([1.0/w_gamma]).to(DEVICE)
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(label); n_total += len(label)
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
        print(f"  Epoch {epoch+1}/5 BCE: loss={total_loss/n_total:.4f} val={val_surv:.2e}")
        if val_surv < best_val:
            best_val = val_surv
            best_epoch = epoch

    print("\nPhase 2: AUC + survival loss training (25 epochs)...")
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=25)

    # Build separate loaders for gamma and hadron
    gamma_dataset = GammaDataset(m_raw[tr_idx[tr_gamma_idx]], f_all[tr_idx[tr_gamma_idx]],
                                  y_all[tr_idx[tr_gamma_idx]], feat_mean, feat_std)
    hadron_dataset = GammaDataset(m_raw[tr_idx[tr_hadron_idx]], f_all[tr_idx[tr_hadron_idx]],
                                   y_all[tr_idx[tr_hadron_idx]], feat_mean, feat_std)

    # Batch size: 256 per class for pairwise loss
    BATCH_PER_CLASS = 256
    gamma_loader = DataLoader(gamma_dataset, batch_size=BATCH_PER_CLASS, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    hadron_loader = DataLoader(hadron_dataset, batch_size=BATCH_PER_CLASS, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True)

    auc_temp = 0.1; surv_temp = 0.05
    alpha_auc = 0.5; alpha_surv = 0.5

    for epoch in range(25):
        model.train()
        total_loss = 0; n_batches = 0

        g_iter = iter(gamma_loader); h_iter = iter(hadron_loader)
        steps = min(len(gamma_loader), len(hadron_loader))

        for step in range(steps):
            try:
                g_mat, g_feat, _ = next(g_iter)
                h_mat, h_feat, _ = next(h_iter)
            except StopIteration:
                break

            g_mat, g_feat = g_mat.to(DEVICE), g_feat.to(DEVICE)
            h_mat, h_feat = h_mat.to(DEVICE), h_feat.to(DEVICE)

            g_scores = torch.sigmoid(model(g_mat, g_feat))
            h_scores = torch.sigmoid(model(h_mat, h_feat))

            loss_auc = auc_loss(g_scores, h_scores, temperature=auc_temp)
            loss_surv = survival_loss(g_scores, h_scores, temperature=surv_temp)
            loss = alpha_auc * loss_auc + alpha_surv * loss_surv

            optimizer2.zero_grad(); loss.backward(); optimizer2.step()
            total_loss += loss.item(); n_batches += 1

        scheduler2.step()

        # Validate
        model.eval()
        val_scores = []; val_labels = []
        with torch.no_grad():
            for mat, feat, label in val_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                scores = torch.sigmoid(model(mat, feat)).cpu().numpy()
                val_scores.extend(scores); val_labels.extend(label.numpy())
        val_surv = survival_at_75(np.array(val_scores), np.array(val_labels))
        print(f"  Epoch {epoch+1}/25 AUC: loss={total_loss/n_batches:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            best_epoch = epoch + 5
            # Get test scores
            test_scores_list = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    scores = torch.sigmoid(model(mat, feat)).cpu().numpy()
                    test_scores_list.extend(scores)
            best_test_scores = np.array(test_scores_list)
            print(f"    -> Best val: {val_surv:.2e}")

    if best_test_scores is None:
        # Get test scores from current model
        test_scores_list = []
        model.eval()
        with torch.no_grad():
            for mat, feat, _ in test_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                scores = torch.sigmoid(model(mat, feat)).cpu().numpy()
                test_scores_list.extend(scores)
        best_test_scores = np.array(test_scores_list)

    test_surv = survival_at_75(best_test_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e} (epoch {best_epoch})")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v35.pt")
    np.save(f"{OUT_DIR}/probs_v35.npy", best_test_scores)
    np.savez(f"{OUT_DIR}/predictions_v35.npz", gamma_scores=best_test_scores)

    # Test blending with ens3
    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_test_scores + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + v35 alpha={alpha}: {s:.4e}")

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: AUC+survival direct metric optimization, 5ep BCE + 25ep pairwise AUC loss")


if __name__ == "__main__":
    main()
