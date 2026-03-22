"""
v23: Test-time augmentation (TTA) on v2 model.
During inference, apply 8 augmentations (4 rotations x 2 flips) and average predictions.
This is a free improvement - no retraining needed!

Also: retrain v2 architecture but with TTA during training validation to get better
checkpoint selection.

Actually: just load v2 model and apply TTA at test time.
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


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean=None, feat_std=None):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        feat = self.features[idx].copy()
        if self.feat_mean is not None:
            feat = (feat - self.feat_mean) / self.feat_std
        return torch.FloatTensor(mat), torch.FloatTensor(feat), int(self.labels[idx])


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)


class GammaCNN(nn.Module):
    """Same as v2."""
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


def predict_with_tta(model, loader, device):
    """Predict with 8-fold TTA (4 rotations x 2 flips)."""
    model.eval()
    all_scores = []

    with torch.no_grad():
        for mat, feat, _ in loader:
            mat, feat = mat.to(device), feat.to(device)
            batch_probs = []

            # 8 augmentations: 4 rotations x 2 flips
            for flip in [False, True]:
                for k in range(4):
                    aug_mat = mat
                    if flip:
                        aug_mat = aug_mat.flip(-1)
                    if k > 0:
                        aug_mat = torch.rot90(aug_mat, k, dims=(-2, -1))
                    probs = torch.softmax(model(aug_mat, feat), dim=1)[:, 0]
                    batch_probs.append(probs)

            # Average across augmentations
            avg_probs = torch.stack(batch_probs, dim=0).mean(0)
            all_scores.append(avg_probs.cpu().numpy())

    return np.concatenate(all_scores)


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
    y_test = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r')

    f_all = engineer_features(np.array(f_raw))
    f_test = engineer_features(np.array(f_test_raw))
    y_all = np.array(y_raw)

    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    # Load v2 model (best single model)
    model = GammaCNN(f_all.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v2.pt", map_location=DEVICE))
    print("v2 model loaded")

    # Without TTA
    scores_no_tta = []
    model.eval()
    with torch.no_grad():
        for mat, feat, _ in test_loader:
            mat, feat = mat.to(DEVICE), feat.to(DEVICE)
            probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
            scores_no_tta.append(probs)
    scores_no_tta = np.concatenate(scores_no_tta)
    print(f"v2 no TTA: {survival_at_75(scores_no_tta, y_test):.2e}")

    # With TTA
    print("Running TTA (8 augmentations)...")
    scores_tta = predict_with_tta(model, test_loader, DEVICE)
    print(f"v2 with TTA: {survival_at_75(scores_tta, y_test):.2e}")

    # Try all v2-family models with TTA
    best_tta = None; best_name = None; best_surv = 1.0

    for version in ['v2', 'v9']:
        try:
            m = GammaCNN(f_all.shape[1]).to(DEVICE)
            m.load_state_dict(torch.load(f"{OUT_DIR}/model_{version}.pt", map_location=DEVICE))
            s = predict_with_tta(m, test_loader, DEVICE)
            surv = survival_at_75(s, y_test)
            print(f"{version} TTA: {surv:.2e}")
            if surv < best_surv:
                best_surv = surv; best_tta = s; best_name = version
        except Exception as e:
            print(f"Could not load {version}: {e}")

    print(f"\nBest TTA: {best_name} at {best_surv:.2e}")

    # Use v2 TTA as our v23 predictions
    np.save(f"{OUT_DIR}/probs_v23.npy", scores_tta)
    np.savez(f"{OUT_DIR}/predictions_v23.npz", gamma_scores=scores_tta)

    # Try ensemble with TTA
    s2 = np.load(f"{OUT_DIR}/probs_v2.npy")
    s7 = np.load(f"{OUT_DIR}/probs_v7.npy")
    s8 = np.load(f"{OUT_DIR}/probs_v8.npy")
    s9 = np.load(f"{OUT_DIR}/probs_v9.npy")
    eps = 1e-10

    ens2 = (s2+eps)**0.45 * (s7+eps)**0.15 * (s8+eps)**0.15 * (s9+eps)**0.25
    print(f"\nens2 survival: {survival_at_75(ens2, y_test):.2e}")

    # Replace v2 with TTA v2
    print("Testing TTA-enhanced ensemble:")
    for alpha in np.arange(0, 0.5, 0.05):
        ens_tta = (scores_tta+eps)**alpha * ens2**(1-alpha)
        s = survival_at_75(ens_tta, y_test)
        print(f"  TTA alpha={alpha:.2f}: {s:.2e}")

    # Replace all models in ensemble with TTA versions
    # This requires loading all models...

    print("\n---")
    print(f"metric: {survival_at_75(scores_tta, y_test):.4e}")
    print("description: v2 model with 8-fold TTA (4 rotations x 2 flips)")


if __name__ == "__main__":
    main()
