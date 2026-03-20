"""
v13: Vision Transformer (ViT) for 16x16 detector grid.
Fundamentally different inductive bias from CNNs.
Global attention allows each muon cell to attend to ALL other cells.
This might capture non-local correlations that CNNs miss.

Architecture: Patch-based ViT with small patches (2x2 or 4x4)
"""

import numpy as np
import torch
import torch.nn as nn
import math
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
    Ze_norm = Ze / 30.0
    Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7
    E_norm = (E - 16.0) / 1.0
    Az_rad = np.radians(Az)
    Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze))
    ne_e_ratio = Ne - E
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.out(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ViTGamma(nn.Module):
    def __init__(self, n_feat, patch_size=2, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        # 16x16 grid with 2x2 patches = 64 patches
        # Each patch has patch_size^2 * 2 channels
        n_patches = (16 // patch_size) ** 2
        patch_dim = patch_size * patch_size * 2  # 2 channels

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model))

        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model + 64, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, mat, feat):
        B, C, H, W = mat.shape
        ps = self.patch_size
        # Patchify: (B, C, H, W) -> (B, n_patches, patch_dim)
        x = mat.reshape(B, C, H // ps, ps, W // ps, ps)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // ps) * (W // ps), -1)
        x = self.patch_embed(x)

        # Add CLS token and positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token

        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cls_out, feat_out], dim=1))


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

    n = len(f_all); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]

    print(f"Train: {len(tr_idx):,} Val: {len(val_idx):,}")

    train_ds = GammaDataset(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = GammaDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = ViTGamma(f_all.shape[1], patch_size=2, d_model=128, n_heads=4, n_layers=4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    print(f"Class weight gamma: {w_gamma:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = 1.0; best_scores = None

    for epoch in range(15):
        model.train()
        total_loss = 0; n_total = 0
        for mat, feat, y in train_loader:
            mat, feat, y = mat.to(DEVICE), feat.to(DEVICE), y.to(DEVICE)
            logits = model(mat, feat)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y); n_total += len(y)
        scheduler.step()

        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for mat, feat, y in val_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
                val_scores.append(probs); val_labels.append(y.numpy())
        val_scores = np.concatenate(val_scores); val_labels = np.concatenate(val_labels)
        val_surv = survival_at_75(val_scores, val_labels)
        print(f"Epoch {epoch+1:2d}/15: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            ts = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    ts.append(torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy())
            best_scores = np.concatenate(ts)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v13.pt")
            print(f"  -> Best: {val_surv:.2e}")

    test_surv = survival_at_75(best_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_v13.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v13.npz", gamma_scores=best_scores)

    print("\n---")
    print(f"metric: {best_val:.4e}")
    print("description: ViT 4-layer 128-dim global attention on 2x2 patches + feat MLP, 15 epochs")


if __name__ == "__main__":
    main()
