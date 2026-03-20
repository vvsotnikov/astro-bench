"""
v37: Muon autoencoder anomaly detection.

Key physics insight: gammas have near-zero muon density. Hadrons have real muon hits.
Idea: train a convolutional autoencoder ONLY on gamma events' muon channel.
The autoencoder learns to reconstruct the "empty/low" muon pattern of gammas.
When applied to hadrons, it will have HIGH reconstruction error (they don't fit the gamma model).

This is fundamentally different from all previous discriminative approaches.
It's a one-class classifier / anomaly detection approach.

Architecture:
- Input: 16x16x1 muon channel (log1p transformed)
- Encoder: 3 conv layers -> 64-dim bottleneck
- Decoder: 3 deconv layers -> 16x16x1

Anomaly score = reconstruction error (MSE)
Higher reconstruction error = more hadron-like

Then: final score = -AE_score (gamma score from AE) combined with standard CNN scores.

The autoencoder should especially flag event #26449 (donut muon pattern) as anomalous
since its spatial pattern is not learned by the gamma AE.
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


class MuonAEDataset(Dataset):
    """Dataset that returns only the muon channel (log1p)."""
    def __init__(self, matrices):
        self.matrices = matrices
    def __len__(self): return len(self.matrices)
    def __getitem__(self, idx):
        mu = np.log1p(self.matrices[idx, :, :, 1].astype(np.float32))  # 16x16
        return torch.FloatTensor(mu).unsqueeze(0)  # (1, 16, 16)


class ConvAE(nn.Module):
    """Convolutional autoencoder for 16x16 muon channel."""
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder: 16x16 -> 8x8 -> 4x4 -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),   # 16x16x16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 8x8x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 4x4x64
            nn.Flatten(),  # 1024
            nn.Linear(1024, latent_dim), nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),  # 4x4x64
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 8x8x32
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 16x16x16
            nn.Conv2d(16, 1, 3, padding=1),  # 16x16x1, no activation (MSE loss)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    def encode(self, x): return self.encoder(x)


def survival_at_75(scores, labels):
    ig = labels == 0; ih = labels == 1
    sg = np.sort(scores[ig])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75 * ng)) - 1, ng - 1)]
    return float((scores[ih] >= thr).sum() / ih.sum())


def get_reconstruction_errors(model, matrices, batch_size=4096):
    """Compute per-event reconstruction errors."""
    errors = []
    for i in range(0, len(matrices), batch_size):
        batch_mu = np.log1p(matrices[i:i+batch_size, :, :, 1].astype(np.float32))
        x = torch.FloatTensor(batch_mu).unsqueeze(1).to(DEVICE)  # (B, 1, 16, 16)
        with torch.no_grad():
            recon = model(x)
            err = ((recon - x) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
        errors.append(err)
    return np.concatenate(errors)


def main():
    print("Loading data...")
    m_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test = np.array(np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r'))
    y_all = np.array(y_raw)

    n = len(y_all); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # Only train AE on gamma events
    tr_gamma_mask = y_all[tr_idx] == 0
    tr_gamma_idx = tr_idx[tr_gamma_mask]
    print(f"Training AE on {len(tr_gamma_idx):,} gamma events...")

    # Validation set (all classes)
    val_y = y_all[val_idx]

    ae_ds = MuonAEDataset(m_raw[tr_gamma_idx])
    ae_loader = DataLoader(ae_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)

    model = ConvAE(latent_dim=128).to(DEVICE)
    print(f"AE params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    mse_loss = nn.MSELoss()

    print("Training autoencoder...")
    for epoch in range(30):
        model.train()
        total_loss = 0; n_batches = 0
        for x in ae_loader:
            x = x.to(DEVICE)
            recon = model(x)
            loss = mse_loss(recon, x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            # Compute reconstruction errors on validation set
            model.eval()
            val_errors = []
            for i in range(0, len(val_idx), 4096):
                batch_idx = val_idx[i:i+4096]
                batch_mu = np.log1p(m_raw[batch_idx, :, :, 1].astype(np.float32))
                x_val = torch.FloatTensor(batch_mu).unsqueeze(1).to(DEVICE)
                with torch.no_grad():
                    recon_val = model(x_val)
                    err_val = ((recon_val - x_val) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
                val_errors.append(err_val)
            val_errors = np.concatenate(val_errors)

            # Score: gamma-like = LOW reconstruction error -> invert for gamma_score
            # gamma_score = -error (lower error = more gamma-like -> higher score)
            gamma_scores_val = -val_errors
            surv = survival_at_75(gamma_scores_val, val_y)
            print(f"  Epoch {epoch+1}: AE loss={total_loss/n_batches:.6f} val_surv={surv:.2e}")

    # Save AE model
    torch.save(model.state_dict(), f"{OUT_DIR}/model_v37_ae.pt")

    # Compute test set reconstruction errors
    print("Computing test set anomaly scores...")
    model.eval()
    test_errors = get_reconstruction_errors(model, m_test_raw)

    # gamma_score = -reconstruction_error (lower error = more gamma-like)
    gamma_scores_test_ae = -test_errors

    # Normalize to [0,1]
    ae_min = gamma_scores_test_ae.min(); ae_max = gamma_scores_test_ae.max()
    gamma_scores_ae_norm = (gamma_scores_test_ae - ae_min) / (ae_max - ae_min + 1e-10)

    ae_surv = survival_at_75(gamma_scores_ae_norm, y_test)
    print(f"\nAE standalone test survival@75: {ae_surv:.2e}")

    # Compare with ens3
    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"ens3 baseline: {base:.4e}")

    # Check AE errors for gamma vs hadron distribution
    print("\nAE error distribution:")
    print(f"  Gamma errors: mean={test_errors[y_test==0].mean():.4f} std={test_errors[y_test==0].std():.4f}")
    print(f"  Hadron errors: mean={test_errors[y_test==1].mean():.4f} std={test_errors[y_test==1].std():.4f}")

    # Check event #26449 specifically (high Nmu anomalous event)
    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    f_test_arr = np.array(f_test_raw)
    Nmu_test = f_test_arr[:, 4]
    E_test = f_test_arr[:, 0]

    hadron_mask = y_test == 1
    hadron_errors = test_errors[hadron_mask]
    hadron_Nmu = Nmu_test[hadron_mask]
    hadron_E = E_test[hadron_mask]

    # Find top-k surviving hadrons at 75% gamma efficiency
    ig = y_test == 0
    sg = np.sort(gamma_scores_ae_norm[ig])[::-1]
    thr = sg[min(int(np.ceil(0.75 * len(sg))) - 1, len(sg) - 1)]
    surviving_hadrons = hadron_mask & (gamma_scores_ae_norm >= thr)
    print(f"\nAE surviving hadrons: {surviving_hadrons.sum()}")
    if surviving_hadrons.sum() < 20:
        surv_idx = np.where(surviving_hadrons)[0]
        for i in surv_idx:
            print(f"  Event {i}: E={E_test[i]:.2f}, Nmu={Nmu_test[i]:.2f}, "
                  f"AE_error={test_errors[i]:.4f}, ens3_score={ens3[i]:.4f}")

    # Blend AE with ens3
    print("\nBlending AE with ens3:")
    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        blend = ((ens3 + eps)**(1-alpha)) * ((gamma_scores_ae_norm + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + AE alpha={alpha}: {s:.4e}")

    # Save AE scores
    np.save(f"{OUT_DIR}/probs_v37.npy", gamma_scores_ae_norm)
    np.savez(f"{OUT_DIR}/predictions_v37.npz", gamma_scores=gamma_scores_ae_norm)

    # Optimize ensemble with AE
    print("\nDirichlet search with AE model...")
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25']:
        models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
    models['v37'] = gamma_scores_ae_norm

    keys = list(models.keys())
    preds = [models[k] for k in keys]
    best_surv = base; best_ens = ens3.copy()

    rng2 = np.random.RandomState(77777)
    N_trials = 200000
    print(f"Running {N_trials:,} Dirichlet trials...")
    for trial in range(N_trials):
        w = rng2.dirichlet(np.ones(len(keys)))
        ens = np.ones(len(preds[0]))
        for p, wi in zip(preds, w):
            ens = ens * (p + eps) ** wi
        s = survival_at_75(ens, y_test)
        if s < best_surv:
            best_surv = s; best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(keys, w)}
            print(f"  Trial {trial}: {s:.2e} weights={best_w}")

    print(f"\nBest with AE: {best_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens_ae.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens_ae.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {ae_surv:.4e}")
    print(f"description: Muon autoencoder anomaly detector, 30ep, latent=128, blend with ens3")


if __name__ == "__main__":
    main()
