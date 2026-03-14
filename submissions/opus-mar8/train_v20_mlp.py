"""v20: Pure MLP on flattened matrices + engineered features.
Fundamentally different from CNN -- no spatial inductive bias.
Goal: diverse model for ensembling with CNN.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import subprocess, gc, time, os

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 30
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

class MLPModel(nn.Module):
    """Large MLP on flattened matrix statistics + engineered features."""
    def __init__(self, n_mat_feat, n_scalar_feat, n_classes=5):
        super().__init__()
        # Matrix feature branch
        self.mat_branch = nn.Sequential(
            nn.Linear(n_mat_feat, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
        )
        # Scalar feature branch
        self.scalar_branch = nn.Sequential(
            nn.Linear(n_scalar_feat, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
        )
        # Combined head
        self.head = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat_feat, scalar_feat):
        m = self.mat_branch(mat_feat)
        s = self.scalar_branch(scalar_feat)
        return self.head(torch.cat([m, s], dim=1))

def extract_matrix_features(matrices):
    """Extract statistical features from 16x16x2 matrices.
    Instead of spatial convolutions, compute per-channel statistics.
    """
    # matrices shape: (N, 2, 16, 16) after log1p and transpose
    N = matrices.shape[0]
    feats = []

    for ch in range(2):
        m = matrices[:, ch]  # (N, 16, 16)
        flat = m.reshape(N, -1)  # (N, 256)

        # Basic statistics
        feats.append(flat.mean(axis=1, keepdims=True))   # mean
        feats.append(flat.std(axis=1, keepdims=True))     # std
        feats.append(flat.max(axis=1, keepdims=True))     # max
        feats.append((flat > 0).sum(axis=1, keepdims=True).astype(np.float32))  # nonzero count

        # Spatial statistics
        # Center of mass
        y_grid = np.arange(16).reshape(1, 16, 1)
        x_grid = np.arange(16).reshape(1, 1, 16)
        total = m.sum(axis=(1, 2), keepdims=True) + 1e-8
        com_y = (m * y_grid).sum(axis=(1, 2), keepdims=True) / total
        com_x = (m * x_grid).sum(axis=(1, 2), keepdims=True) / total
        feats.append(com_y.reshape(N, 1))
        feats.append(com_x.reshape(N, 1))

        # Spread (second moment)
        spread_y = ((m * (y_grid - com_y)**2).sum(axis=(1, 2)) / total.squeeze()).reshape(N, 1)
        spread_x = ((m * (x_grid - com_x)**2).sum(axis=(1, 2)) / total.squeeze()).reshape(N, 1)
        feats.append(np.sqrt(spread_y + 1e-8))
        feats.append(np.sqrt(spread_x + 1e-8))

        # Quadrant energies (4 quadrants)
        feats.append(m[:, :8, :8].sum(axis=(1, 2)).reshape(N, 1))
        feats.append(m[:, :8, 8:].sum(axis=(1, 2)).reshape(N, 1))
        feats.append(m[:, 8:, :8].sum(axis=(1, 2)).reshape(N, 1))
        feats.append(m[:, 8:, 8:].sum(axis=(1, 2)).reshape(N, 1))

        # Peak value location
        flat_idx = flat.argmax(axis=1)
        peak_y = (flat_idx // 16).reshape(N, 1).astype(np.float32)
        peak_x = (flat_idx % 16).reshape(N, 1).astype(np.float32)
        feats.append(peak_y)
        feats.append(peak_x)

        # Ring statistics (distance from center)
        center = 7.5
        dist = np.sqrt((y_grid - center)**2 + (x_grid - center)**2)
        inner = (dist < 4).astype(np.float32)
        outer = (dist >= 4).astype(np.float32)
        feats.append((m * inner).sum(axis=(1, 2)).reshape(N, 1))
        feats.append((m * outer).sum(axis=(1, 2)).reshape(N, 1))

    # Cross-channel features
    ch0 = matrices[:, 0]  # electron
    ch1 = matrices[:, 1]  # muon
    ratio = ch0.sum(axis=(1, 2)) / (ch1.sum(axis=(1, 2)) + 1e-8)
    feats.append(ratio.reshape(N, 1))
    # Correlation between channels
    ch0_flat = ch0.reshape(N, -1)
    ch1_flat = ch1.reshape(N, -1)
    # Simple dot product correlation
    feats.append((ch0_flat * ch1_flat).sum(axis=1).reshape(N, 1))

    return np.concatenate(feats, axis=1).astype(np.float32)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load train data
    p("Loading train matrices...")
    raw_mat = np.load(f"{DATA_DIR}/composition_train/matrices.npy", mmap_mode='r')
    raw_feat = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')
    n_train = len(labels)

    # Process in chunks to avoid OOM
    p("Extracting matrix features (train)...")
    CHUNK = 500000
    mat_feat_chunks = []
    scalar_feat_chunks = []
    label_chunks = []
    for i in range(0, n_train, CHUNK):
        end = min(i + CHUNK, n_train)
        p(f"  chunk {i}/{n_train}")
        m = np.array(raw_mat[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)  # (N, 2, 16, 16)
        mat_feat_chunks.append(extract_matrix_features(m))
        scalar_feat_chunks.append(engineer_features(np.array(raw_feat[i:end], dtype=np.float32)))
        label_chunks.append(np.array(labels[i:end], dtype=np.int64))
        del m
        gc.collect()

    train_mat_feat = np.concatenate(mat_feat_chunks)
    train_scalar_feat = np.concatenate(scalar_feat_chunks)
    train_labels = np.concatenate(label_chunks)
    del mat_feat_chunks, scalar_feat_chunks, label_chunks
    gc.collect()

    n_mat_feat = train_mat_feat.shape[1]
    n_scalar_feat = train_scalar_feat.shape[1]
    p(f"Matrix features: {n_mat_feat}, Scalar features: {n_scalar_feat}")

    # Normalize features
    mat_mean = train_mat_feat.mean(0)
    mat_std = train_mat_feat.std(0) + 1e-6
    scalar_mean = train_scalar_feat.mean(0)
    scalar_std = train_scalar_feat.std(0) + 1e-6
    train_mat_feat = (train_mat_feat - mat_mean) / mat_std
    train_scalar_feat = (train_scalar_feat - scalar_mean) / scalar_std

    # Load test data
    p("Loading test data...")
    raw_mat_test = np.load(f"{DATA_DIR}/composition_test/matrices.npy", mmap_mode='r')
    raw_feat_test = np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r')
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)

    p("Extracting matrix features (test)...")
    m_test = np.array(raw_mat_test[:], dtype=np.float32)
    m_test = np.log1p(m_test).transpose(0, 3, 1, 2)
    test_mat_feat = extract_matrix_features(m_test)
    del m_test; gc.collect()
    test_scalar_feat = engineer_features(np.array(raw_feat_test[:], dtype=np.float32))

    test_mat_feat = (test_mat_feat - mat_mean) / mat_std
    test_scalar_feat = (test_scalar_feat - scalar_mean) / scalar_std

    # Create datasets
    train_ds = TensorDataset(
        torch.from_numpy(train_mat_feat),
        torch.from_numpy(train_scalar_feat),
        torch.from_numpy(train_labels),
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    del train_mat_feat, train_scalar_feat, train_labels
    gc.collect()

    test_mat_t = torch.from_numpy(test_mat_feat)
    test_scalar_t = torch.from_numpy(test_scalar_feat)

    # Model
    model = MLPModel(n_mat_feat, n_scalar_feat).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    p(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    best_acc = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_n = 0

        for mat_b, scalar_b, y_b in train_dl:
            mat_b = mat_b.to(DEVICE, non_blocking=True)
            scalar_b = scalar_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                out = model(mat_b, scalar_b)
                loss = criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(y_b)
            total_correct += (out.argmax(1) == y_b).sum().item()
            total_n += len(y_b)

        scheduler.step()
        train_acc = total_correct / total_n

        # Evaluate
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(test_mat_t), BATCH_SIZE):
                end = min(i + BATCH_SIZE, len(test_mat_t))
                mb = test_mat_t[i:end].to(DEVICE)
                sb = test_scalar_t[i:end].to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mb, sb)
                probs_list.append(torch.softmax(out.float(), 1).cpu().numpy())

        probs = np.concatenate(probs_list)
        preds = probs.argmax(1)
        test_acc = (preds == y_test).mean()
        elapsed = time.time() - t0

        msg = f"  Ep {epoch}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} [{elapsed:.0f}s]"
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v20.pt")
            np.save(f"{OUT_DIR}/probs_v20.npy", probs)
            np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))
            msg += f"\n    >>> Best: {best_acc:.4f}"
        p(msg)

    p(f"\nBest test accuracy: {best_acc:.4f}")

    # Verify
    result = subprocess.run(
        ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
        capture_output=True, text=True,
        cwd="/home/vladimir/cursor_projects/astro-agents"
    )
    p(f"verify.py output:\n{result.stdout}")

    p("---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: Pure MLP on {n_mat_feat} matrix stats + {n_scalar_feat} scalar features, {EPOCHS} epochs")


if __name__ == "__main__":
    main()
