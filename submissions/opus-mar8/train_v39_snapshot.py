"""v39: Snapshot ensemble -- train CNN with cosine annealing warm restarts,
save model at end of each cycle, ensemble the snapshots.
Uses v8 architecture, 5 cycles of 4 epochs each = 20 epochs total.
Each restart explores a different local minimum -> diverse ensemble."""
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
BATCH_SIZE = 4096
CYCLES = 5
EPOCHS_PER_CYCLE = 4
TOTAL_EPOCHS = CYCLES * EPOCHS_PER_CYCLE
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


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class CNNHybrid(nn.Module):
    def __init__(self, n_feat=13, n_classes=5):
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
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
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


def load_all_f32(split, feat_stats=None):
    p(f"Loading {split}...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    mat_list = []
    for i in range(0, n, 250000):
        end = min(i + 250000, n)
        m = np.array(matrices[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
        if (i // 250000) % 5 == 0:
            p(f"  {split}: {end}/{n}")
    mat_tensor = torch.cat(mat_list, dim=0)
    del mat_list; gc.collect()

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
    feat_tensor = torch.from_numpy(feats); del feats; gc.collect()
    label_tensor = torch.from_numpy(np.array(labels[:], dtype=np.int64))

    p(f"  {split}: {mat_tensor.shape}")
    return mat_tensor, feat_tensor, label_tensor, (feat_mean, feat_std)


def evaluate(model, test_loader):
    model.eval()
    all_probs = []
    tc, tt = 0, 0
    with torch.no_grad():
        for mat_b, feat_b, label_b in test_loader:
            mat_b, feat_b, label_b = mat_b.to(DEVICE), feat_b.to(DEVICE), label_b.to(DEVICE)
            with autocast(device_type='cuda'):
                out = model(mat_b, feat_b)
            all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
            tc += (out.argmax(1) == label_b).sum().item()
            tt += len(label_b)
    return tc / tt, np.concatenate(all_probs)


def main():
    t0 = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    mat_train, feat_train, y_train, stats = load_all_f32("train")
    mat_test, feat_test, y_test, _ = load_all_f32("test", feat_stats=stats)

    train_loader = DataLoader(TensorDataset(mat_train, feat_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(mat_test, feat_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = CNNHybrid(n_feat=feat_train.shape[1]).to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS_PER_CYCLE, T_mult=1)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    snapshot_probs = []
    best_single_acc = 0
    y_test_np = np.array(y_test[:])

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        correct, total = 0, 0
        for mat_b, feat_b, label_b in train_loader:
            mat_b, feat_b, label_b = mat_b.to(DEVICE), feat_b.to(DEVICE), label_b.to(DEVICE)
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

        test_acc, probs = evaluate(model, test_loader)

        cycle = epoch // EPOCHS_PER_CYCLE
        epoch_in_cycle = epoch % EPOCHS_PER_CYCLE

        p(f"Ep {epoch+1}/{TOTAL_EPOCHS} (c{cycle+1}e{epoch_in_cycle+1}): train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_single_acc:
            best_single_acc = test_acc
            p(f"  >>> Best single: {best_single_acc:.4f}")

        # Save snapshot at end of each cycle (when LR is lowest)
        if epoch_in_cycle == EPOCHS_PER_CYCLE - 1:
            snapshot_probs.append(probs.copy())
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v39_snap{cycle}.pt")
            p(f"  === Snapshot {cycle+1} saved (acc={test_acc:.4f}) ===")

            # Test ensemble of snapshots so far
            if len(snapshot_probs) > 1:
                avg_probs = np.mean(snapshot_probs, axis=0)
                ens_acc = (avg_probs.argmax(1) == y_test_np).mean()
                p(f"  Ensemble of {len(snapshot_probs)} snapshots: acc={ens_acc:.4f}")

    # Final snapshot ensemble
    avg_probs = np.mean(snapshot_probs, axis=0)
    ens_preds = avg_probs.argmax(1)
    ens_acc = (ens_preds == y_test_np).mean()

    # Save
    np.savez(f"{OUT_DIR}/predictions_v39.npz", predictions=ens_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v39.npy", avg_probs)

    # Also save individual snapshot probs for later optimization
    for i, sp in enumerate(snapshot_probs):
        np.save(f"{OUT_DIR}/probs_v39_snap{i}.npy", sp)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best single: {best_single_acc:.4f}, Ensemble: {ens_acc:.4f}")
    p(f"---")
    p(f"metric: {ens_acc:.4f}")
    p(f"description: Snapshot ensemble ({CYCLES} cycles x {EPOCHS_PER_CYCLE}ep), cosine warm restarts")


if __name__ == "__main__":
    main()
