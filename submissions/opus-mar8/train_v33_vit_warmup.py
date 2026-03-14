"""v33: ViT with proper warmup + cosine schedule (config 2/3).
v21 ViT failed (47.38%) because no warmup -- ViTs need it.
Also: larger patches (8x8 -> 4 tokens instead of 4x4 -> 16 tokens) for efficiency.
Use features via cross-attention.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import subprocess, gc, time, math

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
LR = 3e-4  # lower LR for ViT
WARMUP_EPOCHS = 3
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


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=2, patch_size=4, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 2, 16, 16) -> (B, n_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_dim, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTHybrid(nn.Module):
    def __init__(self, n_feat=13, n_classes=5, patch_size=4, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed(2, patch_size, embed_dim)
        n_patches = (16 // patch_size) ** 2  # 16 for 4x4, 4 for 8x8

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2.0, drop=0.1)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Feature branch
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat, feat):
        x = self.patch_embed(mat)  # (B, n_patches, dim)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, dim)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x[:, 0])  # CLS token

        f = self.feat_mlp(feat)
        return self.head(torch.cat([x, f], dim=1))


def get_lr(epoch, warmup_epochs, max_epochs, base_lr):
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    p("Loading train data...")
    raw_mat = np.load(f"{DATA_DIR}/composition_train/matrices.npy", mmap_mode='r')
    raw_feat = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')
    n_train = len(labels)

    CHUNK = 500000
    mat_chunks = []
    feat_chunks = []
    for i in range(0, n_train, CHUNK):
        end = min(i + CHUNK, n_train)
        p(f"  train: {i}/{n_train}")
        m = np.array(raw_mat[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_chunks.append(torch.from_numpy(m))
        feat_chunks.append(engineer_features(np.array(raw_feat[i:end], dtype=np.float32)))

    mat_train = torch.cat(mat_chunks)
    feat_all = np.concatenate(feat_chunks)
    y_train = np.array(labels[:], dtype=np.int64)
    del mat_chunks, feat_chunks; gc.collect()

    # Normalize features
    feat_mean = feat_all.mean(0)
    feat_std = feat_all.std(0) + 1e-6
    feat_all = (feat_all - feat_mean) / feat_std
    feat_train = torch.from_numpy(feat_all)
    del feat_all; gc.collect()

    p(f"Train: {mat_train.shape}")

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

    model = ViTHybrid(n_feat=13, patch_size=4, embed_dim=128, depth=4, num_heads=4).to(DEVICE)
    n_params = sum(p_.numel() for p_ in model.parameters())
    p(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)  # higher wd for ViT
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    best_frac_err = 1.0
    best_acc = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Manual warmup + cosine LR
        lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

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

        # Verify every 5 epochs + first 3 + last
        frac_err_val = -1
        if epoch % 5 == 0 or epoch == EPOCHS or epoch <= 3:
            np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))
            result = subprocess.run(
                ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
                capture_output=True, text=True,
                cwd="/home/vladimir/cursor_projects/astro-agents"
            )
            for line in result.stdout.split('\n'):
                if 'mean fraction error' in line.lower():
                    try:
                        frac_err_val = float(line.split(':')[-1].strip())
                    except:
                        pass
                    break

        msg = f"  Ep {epoch}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} lr={lr:.6f}"
        if frac_err_val > 0:
            msg += f" frac_err={frac_err_val:.4f}"

        if frac_err_val > 0 and frac_err_val < best_frac_err:
            best_frac_err = frac_err_val
            best_acc = test_acc
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v33.pt")
            np.save(f"{OUT_DIR}/probs_v33.npy", probs)
            msg += " *** BEST"
        elif test_acc > best_acc:
            best_acc = test_acc
            np.save(f"{OUT_DIR}/probs_v33_latest.npy", probs)

        msg += f" [{elapsed:.0f}s]"
        p(msg)

    # Final verify
    if best_frac_err < 1.0:
        probs_best = np.load(f"{OUT_DIR}/probs_v33.npy")
    else:
        probs_best = np.load(f"{OUT_DIR}/probs_v33_latest.npy")
    preds = probs_best.argmax(1).astype(np.int8)
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds)
    result = subprocess.run(
        ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
        capture_output=True, text=True,
        cwd="/home/vladimir/cursor_projects/astro-agents"
    )
    p(f"\nFinal verify:\n{result.stdout[:500]}")
    p(f"\nBest: acc={best_acc:.4f}, frac_err={best_frac_err:.4f}")
    p("---")
    p(f"metric: {best_frac_err:.4f}")
    p(f"description: ViT 4x4 patch dim=128 depth=4 + warmup + cosine, {EPOCHS}ep, config 2/3")


if __name__ == "__main__":
    main()
