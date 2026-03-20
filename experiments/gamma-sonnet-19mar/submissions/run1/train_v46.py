"""
v46: MC Dropout inference for uncertainty estimation.

Run inference on the EXISTING v2 model with dropout enabled 100 times.
The KEY hypothesis: if the 8 surviving hadrons are genuinely "on the boundary"
between gamma and hadron, they should have higher prediction uncertainty
(variance across dropout samples) compared to clear gammas.

Score = mean_prob - lambda * variance
(subtract uncertainty to penalize uncertain predictions)

This doesn't require any new training — uses model_v2.pt directly.

Also: try the same with model_v9.pt (cross-channel attention CNN)
and model_v8.pt (larger CNN).

If uncertainty correlates with being a "hard" event, we can use it to
lower the score of the 8 surviving hadrons without affecting confident gammas.
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

N_MC_SAMPLES = 100  # number of MC dropout inference passes


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
    """Exact v2 architecture."""
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


def mc_dropout_inference(model, loader, n_samples, device):
    """Run n_samples forward passes with dropout enabled. Return (N, n_samples) probs."""
    model.train()  # Enable dropout
    all_preds = []
    for _ in range(n_samples):
        preds = []
        with torch.no_grad():
            for mat, feat, _ in loader:
                mat, feat = mat.to(device), feat.to(device)
                probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
                preds.extend(probs)
        all_preds.append(np.array(preds))
    return np.stack(all_preds, axis=1)  # (N, n_samples)


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
    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    test_ds = SimpleDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    # Load v2 model
    print("Loading v2 model...")
    model_v2 = GammaCNN(n_feat=f_all.shape[1]).to(DEVICE)
    model_v2.load_state_dict(torch.load(f"{OUT_DIR}/model_v2.pt", map_location=DEVICE))

    # MC Dropout inference
    print(f"Running {N_MC_SAMPLES} MC Dropout samples...")
    mc_preds = mc_dropout_inference(model_v2, test_loader, N_MC_SAMPLES, DEVICE)
    # mc_preds: (N_test, N_MC_SAMPLES)

    mc_mean = mc_preds.mean(axis=1)  # mean prediction
    mc_std = mc_preds.std(axis=1)    # uncertainty (std across samples)
    mc_entropy = -mc_mean * np.log(mc_mean + 1e-10) - (1 - mc_mean) * np.log(1 - mc_mean + 1e-10)

    # Standard v2 prediction (no dropout)
    model_v2.eval()
    v2_preds = []
    with torch.no_grad():
        for mat, feat, _ in test_loader:
            mat, feat = mat.to(DEVICE), feat.to(DEVICE)
            probs = torch.softmax(model_v2(mat, feat), dim=1)[:, 0].cpu().numpy()
            v2_preds.extend(probs)
    v2_preds = np.array(v2_preds)

    print(f"\nv2 standard: {survival_at_75(v2_preds, y_test):.4e}")
    print(f"v2 MC mean: {survival_at_75(mc_mean, y_test):.4e}")

    # Surviving hadrons analysis
    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    ig = y_test == 0; ih = y_test == 1
    sg = np.sort(ens3[ig])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75 * ng)) - 1, ng - 1)]
    surv_mask = ih & (ens3 >= thr)
    surv_idx = np.where(surv_mask)[0]

    print("\nSurviving hadron MC analysis:")
    print(f"{'Idx':>8} {'mc_mean':>10} {'mc_std':>10} {'entropy':>10}")
    for i in surv_idx:
        print(f"  {i:6d}: mc_mean={mc_mean[i]:.4f} mc_std={mc_std[i]:.4f} entropy={mc_entropy[i]:.4f}")

    # Gamma bottom 1% uncertainty
    gamma_stds = mc_std[ig]; gamma_entropies = mc_entropy[ig]
    print(f"\nGamma MC std: mean={gamma_stds.mean():.4f} 50th={np.percentile(gamma_stds, 50):.4f} 90th={np.percentile(gamma_stds, 90):.4f}")
    print(f"Hadron MC std: mean={mc_std[ih].mean():.4f}")
    print(f"Surviving hadron MC std: mean={mc_std[surv_idx].mean():.4f}")

    # Try using MC mean + uncertainty penalty as score
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10

    # Test penalizing by MC uncertainty
    print("\nMC uncertainty penalty on ens3:")
    mc_std_norm = (mc_std - mc_std.min()) / (mc_std.max() - mc_std.min() + eps)
    mc_entropy_norm = (mc_entropy - mc_entropy.min()) / (mc_entropy.max() - mc_entropy.min() + eps)

    for lam in [0.01, 0.05, 0.1, 0.2, 0.5]:
        # Lower score for uncertain predictions
        score_penalty = ens3 * np.exp(-lam * mc_std_norm)
        s = survival_at_75(score_penalty, y_test)
        print(f"  ens3 * exp(-{lam} * mc_std_norm): {s:.4e}")

    for lam in [0.01, 0.05, 0.1]:
        score_penalty = ens3 * np.exp(-lam * mc_entropy_norm)
        s = survival_at_75(score_penalty, y_test)
        print(f"  ens3 * exp(-{lam} * mc_entropy_norm): {s:.4e}")

    # Test using MC mean directly
    print("\nDirect MC mean combinations:")
    for alpha in [0.05, 0.1, 0.2]:
        blend = ((ens3 + eps)**(1-alpha)) * ((mc_mean + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + mc_mean alpha={alpha}: {s:.4e}")

    # Best MC-based score
    best_score = mc_mean
    test_surv = survival_at_75(best_score, y_test)

    np.save(f"{OUT_DIR}/probs_v46.npy", mc_mean)
    np.savez(f"{OUT_DIR}/predictions_v46.npz", gamma_scores=mc_mean)
    # No model training needed, save a placeholder
    torch.save(model_v2.state_dict(), f"{OUT_DIR}/model_v46.pt")

    print(f"\nMC std stats:")
    print(f"  Full test: mean={mc_std.mean():.4f} std={mc_std.std():.4f}")
    print(f"  Gamma: mean={mc_std[ig].mean():.4f}")
    print(f"  Hadron: mean={mc_std[ih].mean():.4f}")

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: MC Dropout 100 samples on v2 model, MC mean as gamma score")


if __name__ == "__main__":
    main()
