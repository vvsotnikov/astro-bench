"""v4: Spatial feature extraction + CNN.
Key insight: different primaries produce different SPATIAL distributions.
- Protons: narrow, more peaked shower
- Iron: broader, more uniform shower (56 sub-showers)
Extract explicit spatial statistics from matrices and feed them as features.
Also try per-pixel augmentation for better diversity.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import time
import gc

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-composition-mar14"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS = 25
LR = 1e-3
LABEL_SMOOTH = 0.05
SEED = 7  # different seed from v8 (42)

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    feats = [
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu,
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
    ]
    return np.stack(feats, axis=1).astype(np.float32)


def extract_spatial_features(matrices_np):
    """Extract spatial statistics from matrices (N, H, W, 2)."""
    n = len(matrices_np)
    chunk = 50000
    all_feats = []
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        m = matrices_np[i:end].astype(np.float32)
        ch0 = m[:, :, :, 0]  # electron/photon
        ch1 = m[:, :, :, 1]  # muon

        feats_list = []
        for ch in [ch0, ch1]:
            # Total sum
            total = ch.sum(axis=(1, 2))
            # Max value
            max_val = ch.max(axis=(1, 2))
            # Non-zero fraction
            nnz = (ch > 0).sum(axis=(1, 2)).astype(np.float32) / 256
            # Center of mass (weighted mean of coordinates)
            yy, xx = np.mgrid[0:16, 0:16]
            total_safe = total.copy()
            total_safe[total_safe == 0] = 1
            cx = (ch * xx[None]).sum(axis=(1, 2)) / total_safe
            cy = (ch * yy[None]).sum(axis=(1, 2)) / total_safe
            # Spread (weighted std of coordinates)
            spread_x = np.sqrt(np.abs((ch * (xx[None] - cx[:, None, None])**2).sum(axis=(1, 2)) / total_safe))
            spread_y = np.sqrt(np.abs((ch * (yy[None] - cy[:, None, None])**2).sum(axis=(1, 2)) / total_safe))
            spread = (spread_x + spread_y) / 2
            # Kurtosis (peakedness)
            r2 = (xx[None] - cx[:, None, None])**2 + (yy[None] - cy[:, None, None])**2
            moment4 = (ch * r2**2).sum(axis=(1, 2)) / total_safe
            moment2 = (ch * r2).sum(axis=(1, 2)) / total_safe
            moment2_safe = moment2.copy()
            moment2_safe[moment2_safe == 0] = 1
            kurtosis = moment4 / moment2_safe**2

            feats_list.extend([
                np.log1p(total), np.log1p(max_val), nnz,
                cx / 15, cy / 15,  # normalized center
                spread / 8,  # normalized spread
                kurtosis,
            ])

        # Ratio features
        total0 = np.log1p(ch0.sum(axis=(1, 2)))
        total1 = np.log1p(ch1.sum(axis=(1, 2)))
        feats_list.append(total0 - total1)  # log ratio
        feats_list.append((ch0 > 0).sum(axis=(1, 2)).astype(np.float32) / ((ch1 > 0).sum(axis=(1, 2)).astype(np.float32) + 1))

        all_feats.append(np.stack(feats_list, axis=1))

    return np.concatenate(all_feats)


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class HybridSpatialModel(nn.Module):
    def __init__(self, n_scalar_feat=13, n_spatial_feat=16, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256),
            nn.AdaptiveAvgPool2d(1),
        )
        n_combined = n_scalar_feat + n_spatial_feat
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_combined, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(384, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat, feat):
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))


def load_data(split, feat_stats=None, spatial_stats=None):
    p(f"Loading {split} data...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    # Load matrices as log1p float32
    chunk = 250000
    mat_list = []
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        m = np.array(matrices[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
        if (i // chunk) % 4 == 0:
            p(f"  {split} matrices: {end}/{n}")
    mat_tensor = torch.cat(mat_list, dim=0)
    del mat_list; gc.collect()

    # Extract spatial features
    p(f"  Extracting spatial features...")
    spatial_feats = extract_spatial_features(matrices)
    p(f"  Spatial features: {spatial_feats.shape}")

    # Engineer scalar features
    feat_chunks = []
    for i in range(0, n, 500000):
        end = min(i + 500000, n)
        f = np.array(raw_feats[i:end], dtype=np.float32)
        feat_chunks.append(engineer_features(f))
    scalar_feats = np.concatenate(feat_chunks)

    # Combine scalar + spatial features
    combined = np.concatenate([scalar_feats, spatial_feats], axis=1).astype(np.float32)
    del scalar_feats, spatial_feats; gc.collect()

    if feat_stats is None:
        feat_mean = combined.mean(0)
        feat_std = combined.std(0) + 1e-6
    else:
        feat_mean, feat_std = feat_stats
    combined = (combined - feat_mean) / feat_std
    feat_tensor = torch.from_numpy(combined)
    del combined; gc.collect()

    label_tensor = torch.from_numpy(np.array(labels[:], dtype=np.int64))
    p(f"  {split}: mat={mat_tensor.shape}, feat={feat_tensor.shape}")
    return mat_tensor, feat_tensor, label_tensor, (feat_mean, feat_std)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    mat_train, feat_train, y_train, stats = load_data("train")
    mat_test, feat_test, y_test, _ = load_data("test", feat_stats=stats)

    n_spatial = feat_train.shape[1] - 13  # spatial features beyond scalar
    p(f"Scalar features: 13, Spatial features: {n_spatial}")

    train_loader = DataLoader(
        TensorDataset(mat_train, feat_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(mat_test, feat_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridSpatialModel(n_scalar_feat=13, n_spatial_feat=n_spatial).to(DEVICE)
    p(f"Params: {sum(pp.numel() for pp in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    best_preds = None
    best_probs = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for mat_b, feat_b, label_b in train_loader:
            mat_b = mat_b.to(DEVICE)
            feat_b = feat_b.to(DEVICE)
            label_b = label_b.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                out = model(mat_b, feat_b)
                loss = criterion(out, label_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (out.argmax(1) == label_b).sum().item()
            total += len(label_b)
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_preds, all_probs = [], []
        tc, tt = 0, 0
        with torch.no_grad():
            for mat_b, feat_b, label_b in test_loader:
                mat_b = mat_b.to(DEVICE)
                feat_b = feat_b.to(DEVICE)
                label_b = label_b.to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mat_b, feat_b)
                all_preds.append(out.argmax(1).cpu().numpy())
                all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
                tc += (out.argmax(1) == label_b).sum().item()
                tt += len(label_b)
        test_acc = tc / tt
        lr = optimizer.param_groups[0]['lr']
        p(f"Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} lr={lr:.6f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            best_probs = np.concatenate(all_probs)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v4.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions_v4.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v4.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p("---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: CNN+Attn + SpatialFeat MLP, {EPOCHS}ep, seed={SEED}")


if __name__ == "__main__":
    main()
