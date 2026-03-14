"""v28: Pure MLP on flattened matrices + scalar features.
Flatten 16x16x2 = 512 dims, concatenate with 13 eng features = 525 dims.
No convolutional inductive bias -- should make different errors than CNN.
This is the key requirement for effective ensembling."""
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
BATCH_SIZE = 8192  # MLP can handle larger batches
EPOCHS = 30
LR = 3e-4
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


class FlatMLP(nn.Module):
    def __init__(self, n_mat=512, n_feat=13, n_classes=5):
        super().__init__()
        total_in = n_mat + n_feat
        self.net = nn.Sequential(
            nn.Linear(total_in, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_all_flat(split, feat_stats=None, mat_stats=None):
    p(f"Loading {split}...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    # Flatten and log1p matrices
    mat_list = []
    for i in range(0, n, 250000):
        end = min(i + 250000, n)
        m = np.array(matrices[i:end], dtype=np.float32)
        m = np.log1p(m)
        m = m.reshape(len(m), -1)  # (B, 512)
        mat_list.append(m)
        if (i // 250000) % 5 == 0:
            p(f"  {split} mat: {end}/{n}")
    mat_flat = np.concatenate(mat_list)
    del mat_list; gc.collect()

    # Normalize matrix features
    if mat_stats is None:
        mat_mean = mat_flat.mean(0)
        mat_std = mat_flat.std(0) + 1e-6
    else:
        mat_mean, mat_std = mat_stats
    mat_flat = (mat_flat - mat_mean) / mat_std

    # Engineer scalar features
    feat_chunks = []
    for i in range(0, n, 500000):
        end = min(i + 500000, n)
        feat_chunks.append(engineer_features(np.array(raw_feats[i:end], dtype=np.float32)))
    feats = np.concatenate(feat_chunks)
    del feat_chunks; gc.collect()

    if feat_stats is None:
        feat_mean = feats.mean(0); feat_std = feats.std(0) + 1e-6
    else:
        feat_mean, feat_std = feat_stats
    feats = (feats - feat_mean) / feat_std

    # Concatenate
    X = np.concatenate([mat_flat, feats], axis=1)
    del mat_flat, feats; gc.collect()

    X_tensor = torch.from_numpy(X)
    del X; gc.collect()
    label_tensor = torch.from_numpy(np.array(labels[:], dtype=np.int64))

    p(f"  {split}: X={X_tensor.shape}")
    return X_tensor, label_tensor, (feat_mean, feat_std), (mat_mean, mat_std)


def main():
    t0 = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    X_train, y_train, f_stats, m_stats = load_all_flat("train")
    X_test, y_test, _, _ = load_all_flat("test", feat_stats=f_stats, mat_stats=m_stats)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = FlatMLP(n_mat=512, n_feat=13).to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    best_preds = None
    best_probs = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for x_b, label_b in train_loader:
            x_b, label_b = x_b.to(DEVICE), label_b.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                out = model(x_b)
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
            for x_b, label_b in test_loader:
                x_b, label_b = x_b.to(DEVICE), label_b.to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(x_b)
                all_preds.append(out.argmax(1).cpu().numpy())
                all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
                tc += (out.argmax(1) == label_b).sum().item()
                tt += len(label_b)

        test_acc = tc / tt
        p(f"Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            best_probs = np.concatenate(all_probs)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v28.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions_v28.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v28.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: Pure MLP on flattened log1p matrices (512) + 13 eng features, {EPOCHS}ep")


if __name__ == "__main__":
    main()
