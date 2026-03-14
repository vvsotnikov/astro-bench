"""v42: Rich feature extraction from matrices + deep MLP.
Instead of 2D CNN, extract statistical features from the 16x16x2 matrices:
- Per-channel: mean, std, max, sum, nnz, percentiles
- Spatial: center of mass, spread, radial profile
- Cross-channel: correlation, ratio statistics
Combined with 13 engineered scalar features -> deep MLP.
This is a fundamentally different representation than CNN."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import time
import gc

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 8192
EPOCHS = 30
LR = 1e-3
LABEL_SMOOTH = 0.05

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


def extract_matrix_features(matrices_chunk):
    """Extract rich statistical features from 16x16x2 matrices."""
    n = len(matrices_chunk)
    m = np.log1p(matrices_chunk.astype(np.float32))  # (n, 16, 16, 2)

    ch0 = m[:, :, :, 0]  # electron channel (n, 16, 16)
    ch1 = m[:, :, :, 1]  # muon channel (n, 16, 16)

    features = []

    for ch, name in [(ch0, 'e'), (ch1, 'mu')]:
        flat = ch.reshape(n, -1)  # (n, 256)

        # Basic stats
        features.append(flat.mean(1, keepdims=True))
        features.append(flat.std(1, keepdims=True))
        features.append(flat.max(1, keepdims=True))
        features.append(flat.sum(1, keepdims=True))
        features.append((flat > 0).sum(1, keepdims=True).astype(np.float32))

        # Percentiles (25, 50, 75, 90, 95, 99 of non-zero values handled via full array)
        for pct in [50, 75, 90, 95, 99]:
            features.append(np.percentile(flat, pct, axis=1, keepdims=True))

        # Spatial center of mass
        y_grid, x_grid = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
        y_grid = y_grid.reshape(1, -1).astype(np.float32)
        x_grid = x_grid.reshape(1, -1).astype(np.float32)
        total = flat.sum(1, keepdims=True) + 1e-8
        cx = (flat * x_grid).sum(1, keepdims=True) / total
        cy = (flat * y_grid).sum(1, keepdims=True) / total
        features.append(cx)
        features.append(cy)

        # Spread (spatial std)
        sx = np.sqrt(((flat * (x_grid - cx)**2).sum(1, keepdims=True) / total).clip(0))
        sy = np.sqrt(((flat * (y_grid - cy)**2).sum(1, keepdims=True) / total).clip(0))
        features.append(sx)
        features.append(sy)

        # Radial profile from center (7.5, 7.5)
        r_grid = np.sqrt((x_grid - 7.5)**2 + (y_grid - 7.5)**2).reshape(1, -1)
        for r_max in [3, 5, 7, 10]:
            mask = (r_grid <= r_max).astype(np.float32)
            features.append((flat * mask).sum(1, keepdims=True) / (mask.sum() + 1e-8))

        # Quadrant sums
        for qr in [(0, 8, 0, 8), (0, 8, 8, 16), (8, 16, 0, 8), (8, 16, 8, 16)]:
            q = ch[:, qr[0]:qr[1], qr[2]:qr[3]].reshape(n, -1).sum(1, keepdims=True)
            features.append(q)

    # Cross-channel features
    flat0 = ch0.reshape(n, -1)
    flat1 = ch1.reshape(n, -1)
    features.append((flat0 * flat1).sum(1, keepdims=True))  # dot product
    features.append((flat0.sum(1, keepdims=True) / (flat1.sum(1, keepdims=True) + 1e-8)))  # ratio
    features.append(((flat0 > 0) & (flat1 > 0)).sum(1, keepdims=True).astype(np.float32))  # co-occurrence
    features.append(((flat0 > 0) | (flat1 > 0)).sum(1, keepdims=True).astype(np.float32))  # union
    features.append((flat0 - flat1).mean(1, keepdims=True))  # mean diff
    features.append((flat0 - flat1).std(1, keepdims=True))  # std diff

    return np.concatenate(features, axis=1)


class DeepMLP(nn.Module):
    def __init__(self, n_feat, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    t0 = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # Extract features
    for split in ['train', 'test']:
        p(f"Extracting features for {split}...")
        matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
        raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
        labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
        n = len(labels)

        mat_feat_chunks = []
        eng_feat_chunks = []
        CHUNK = 250000
        for i in range(0, n, CHUNK):
            end = min(i + CHUNK, n)
            mat_feat_chunks.append(extract_matrix_features(np.array(matrices[i:end])))
            eng_feat_chunks.append(engineer_features(np.array(raw_feats[i:end], dtype=np.float32)))
            if i % (CHUNK * 5) == 0:
                p(f"  {split}: {end}/{n}")

        mat_feats = np.concatenate(mat_feat_chunks)
        eng_feats = np.concatenate(eng_feat_chunks)
        all_feats = np.concatenate([mat_feats, eng_feats], axis=1)
        del mat_feat_chunks, eng_feat_chunks; gc.collect()

        p(f"  {split}: {all_feats.shape[1]} features")

        if split == 'train':
            feat_mean = all_feats.mean(0); feat_std = all_feats.std(0) + 1e-6
            all_feats = (all_feats - feat_mean) / feat_std
            feat_train = torch.from_numpy(all_feats.astype(np.float32))
            y_train = torch.from_numpy(np.array(labels[:], dtype=np.int64))
            del all_feats; gc.collect()
        else:
            all_feats = (all_feats - feat_mean) / feat_std
            feat_test = torch.from_numpy(all_feats.astype(np.float32))
            y_test = torch.from_numpy(np.array(labels[:], dtype=np.int64))
            del all_feats; gc.collect()

    p(f"Feature dim: {feat_train.shape[1]}")
    p(f"Data loaded in {time.time()-t0:.0f}s")

    train_loader = DataLoader(TensorDataset(feat_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(feat_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = DeepMLP(n_feat=feat_train.shape[1]).to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    best_probs = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for feat_b, label_b in train_loader:
            feat_b, label_b = feat_b.to(DEVICE), label_b.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                out = model(feat_b)
                loss = criterion(out, label_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (out.argmax(1) == label_b).sum().item()
            total += len(label_b)
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_probs = []
        tc, tt = 0, 0
        with torch.no_grad():
            for feat_b, label_b in test_loader:
                feat_b, label_b = feat_b.to(DEVICE), label_b.to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(feat_b)
                all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
                tc += (out.argmax(1) == label_b).sum().item()
                tt += len(label_b)

        test_acc = tc / tt
        p(f"Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_probs = np.concatenate(all_probs)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v42.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    preds = best_probs.argmax(1).astype(np.int8)
    np.savez(f"{OUT_DIR}/predictions_v42.npz", predictions=preds)
    np.save(f"{OUT_DIR}/probs_v42.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: Deep MLP on rich matrix features ({feat_train.shape[1]}d) + eng feats, {EPOCHS}ep")


if __name__ == "__main__":
    main()
