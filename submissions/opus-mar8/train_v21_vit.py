"""v21: Vision Transformer on 16x16x2 matrices + MLP on scalar features.
Fundamentally different inductive bias from CNN -- global attention from the start.
Config 1: small ViT with 4x4 patches (16 patches), 4 layers, dim=128.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import subprocess, gc, time, os, math

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
LR = 3e-4
EPOCHS = 25
SEED = 42

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    return np.stack([
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu, (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
    ], axis=1).astype(np.float32)


class PatchEmbedding(nn.Module):
    """Split 16x16x2 image into patches and embed them."""
    def __init__(self, patch_size=4, in_channels=2, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (16 // patch_size) ** 2  # 16 for 4x4 patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 2, 16, 16) -> (B, n_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTHybrid(nn.Module):
    """Vision Transformer + scalar feature MLP."""
    def __init__(self, patch_size=4, embed_dim=128, depth=4, n_heads=4,
                 n_feat=13, n_classes=5, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, 2, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim + 256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat, feat):
        B = mat.shape[0]
        x = self.patch_embed(mat)  # (B, n_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token

        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cls_out, feat_out], dim=1))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load ALL training data
    p("Loading train data...")
    raw_mat = np.load(f"{DATA_DIR}/composition_train/matrices.npy", mmap_mode='r')
    raw_feat = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')
    n_train = len(labels)

    CHUNK = 500000
    mat_chunks = []
    feat_chunks = []
    p("Loading and transforming data...")
    for i in range(0, n_train, CHUNK):
        end = min(i + CHUNK, n_train)
        p(f"  {i}/{n_train}")
        m = np.array(raw_mat[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_chunks.append(torch.from_numpy(m))
        feat_chunks.append(engineer_features(np.array(raw_feat[i:end], dtype=np.float32)))

    mat_train = torch.cat(mat_chunks)
    feat_all = np.concatenate(feat_chunks)
    y_train = np.array(labels[:], dtype=np.int64)
    del mat_chunks, feat_chunks; gc.collect()

    feat_mean = feat_all.mean(0)
    feat_std = feat_all.std(0) + 1e-6
    feat_all = (feat_all - feat_mean) / feat_std
    feat_train = torch.from_numpy(feat_all)
    del feat_all; gc.collect()

    p(f"Train: {mat_train.shape}, {feat_train.shape}")

    # Load test
    p("Loading test data...")
    raw_mat_test = np.load(f"{DATA_DIR}/composition_test/matrices.npy", mmap_mode='r')
    raw_feat_test = np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r')
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)

    mat_list = []
    for i in range(0, len(y_test), 250000):
        end = min(i + 250000, len(y_test))
        m = np.array(raw_mat_test[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
    mat_test = torch.cat(mat_list)
    del mat_list; gc.collect()

    test_feats = engineer_features(np.array(raw_feat_test[:], dtype=np.float32))
    test_feats = (test_feats - feat_mean) / feat_std
    feat_test = torch.from_numpy(test_feats)

    # Dataset
    train_ds = TensorDataset(mat_train, feat_train, torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Model
    model = ViTHybrid(patch_size=4, embed_dim=128, depth=4, n_heads=4).to(DEVICE)
    n_params = sum(p_.numel() for p_ in model.parameters())
    p(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    best_acc = 0
    best_frac_err = 1.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_correct = 0
        total_n = 0

        for mat_b, feat_b, y_b in train_dl:
            mat_b = mat_b.to(DEVICE, non_blocking=True)
            feat_b = feat_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                out = model(mat_b, feat_b)
                loss = criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_correct += (out.argmax(1) == y_b).sum().item()
            total_n += len(y_b)

        scheduler.step()
        train_acc = total_correct / total_n

        # Evaluate
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(y_test), BATCH_SIZE):
                end = min(i + BATCH_SIZE, len(y_test))
                mb = mat_test[i:end].to(DEVICE)
                fb = feat_test[i:end].to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mb, fb)
                probs_list.append(torch.softmax(out.float(), 1).cpu().numpy())

        probs = np.concatenate(probs_list)
        preds = probs.argmax(1)
        test_acc = (preds == y_test).mean()
        elapsed = time.time() - t0

        # Run verify.py every epoch
        np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))
        result = subprocess.run(
            ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
            capture_output=True, text=True,
            cwd="/home/vladimir/cursor_projects/astro-agents"
        )
        frac_err_str = "N/A"
        frac_err_val = 1.0
        for line in result.stdout.split('\n'):
            if 'mean fraction error' in line.lower():
                frac_err_str = line.strip()
                try:
                    frac_err_val = float(line.split(':')[-1].strip())
                except:
                    pass
                break

        msg = f"  Ep {epoch}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} frac_err={frac_err_val:.4f} [{elapsed:.0f}s]"

        improved = False
        if frac_err_val < best_frac_err:
            best_frac_err = frac_err_val
            best_acc = test_acc
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v21.pt")
            np.save(f"{OUT_DIR}/probs_v21.npy", probs)
            improved = True
            msg += f" *** BEST frac_err"
        elif test_acc > best_acc and not improved:
            best_acc = test_acc
            msg += f" (best acc)"

        p(msg)

    p(f"\nBest: acc={best_acc:.4f}, frac_err={best_frac_err:.4f}")

    # Restore best predictions
    probs_best = np.load(f"{OUT_DIR}/probs_v21.npy")
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=probs_best.argmax(1).astype(np.int8))

    # Final verify
    result = subprocess.run(
        ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
        capture_output=True, text=True,
        cwd="/home/vladimir/cursor_projects/astro-agents"
    )
    p(f"Final verify:\n{result.stdout[:500]}")

    p("---")
    p(f"metric: {best_frac_err:.4f}")
    p(f"description: ViT (4x4 patch, dim=128, depth=4) + MLP, {EPOCHS}ep, all 5.5M data")


if __name__ == "__main__":
    main()
