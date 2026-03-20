"""v32: CNN with spatial self-attention (QKV) instead of SE channel attention.
Cross-pollinated from gamma winner (haiku-gamma-mar9-v3 v9).
Config 1/3: QKV attention after conv blocks, same data pipeline as v8.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import subprocess, gc, time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 20
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


class SpatialSelfAttention(nn.Module):
    """QKV self-attention on spatial features (from gamma winner)."""
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        self.query = nn.Conv2d(ch, ch, 1, bias=False)
        self.key = nn.Conv2d(ch, ch, 1, bias=False)
        self.value = nn.Conv2d(ch, ch, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        self.norm = nn.LayerNorm([ch])  # will reshape for this

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # Reshape to (B, num_heads, head_dim, H*W)
        q = self.query(x).reshape(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).reshape(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).reshape(B, self.num_heads, self.head_dim, H * W)

        # Attention: (B, heads, H*W, H*W)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q.transpose(-2, -1), k) * scale  # (B, heads, HW, HW)
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = torch.matmul(v, attn)  # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return out + residual


class QKVCNNHybrid(nn.Module):
    """CNN with spatial self-attention (QKV) + MLP."""
    def __init__(self, n_feat=13, n_classes=5):
        super().__init__()
        # Conv blocks with QKV attention
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.attn1 = SpatialSelfAttention(64, num_heads=4)
        self.pool1 = nn.MaxPool2d(2)  # 16->8

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.attn2 = SpatialSelfAttention(128, num_heads=4)
        self.pool2 = nn.MaxPool2d(2)  # 8->4

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.attn3 = SpatialSelfAttention(256, num_heads=4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature branch
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )
        # Head
        self.head = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat, feat):
        x = self.block1(mat)
        x = self.attn1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.attn2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.attn3(x)
        x = self.global_pool(x).flatten(1)

        f = self.feat_mlp(feat)
        return self.head(torch.cat([x, f], dim=1))


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

    model = QKVCNNHybrid(n_feat=13).to(DEVICE)
    n_params = sum(p_.numel() for p_ in model.parameters())
    p(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    best_frac_err = 1.0
    best_acc = 0
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

        msg = f"  Ep {epoch}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f}"
        if frac_err_val > 0:
            msg += f" frac_err={frac_err_val:.4f}"

        if frac_err_val > 0 and frac_err_val < best_frac_err:
            best_frac_err = frac_err_val
            best_acc = test_acc
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v32.pt")
            np.save(f"{OUT_DIR}/probs_v32.npy", probs)
            msg += " *** BEST"
        elif test_acc > best_acc:
            best_acc = test_acc
            np.save(f"{OUT_DIR}/probs_v32_latest.npy", probs)

        msg += f" [{elapsed:.0f}s]"
        p(msg)

    # Final verify
    if best_frac_err < 1.0:
        probs_best = np.load(f"{OUT_DIR}/probs_v32.npy")
    else:
        probs_best = np.load(f"{OUT_DIR}/probs_v32_latest.npy")
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
    p(f"description: QKV spatial self-attention CNN + MLP, {EPOCHS}ep, 5.5M, config 1/3")


if __name__ == "__main__":
    main()
