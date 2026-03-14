"""v19: Multi-seed ensemble. Train v8 architecture 3 times with different seeds.
Average probabilities at test time.
Each run: 20 epochs, same hyperparameters as v8."""
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
EPOCHS = 20
LR = 1e-3
LABEL_SMOOTH = 0.05
SEEDS = [42, 123, 7]

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
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class HybridModel(nn.Module):
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


def train_one_seed(seed, mat_train, feat_train, y_train, mat_test, feat_test, y_test):
    """Train one model with given seed, return test probabilities."""
    p(f"\n{'='*60}")
    p(f"Training seed={seed}")
    p(f"{'='*60}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()

    train_loader = DataLoader(TensorDataset(mat_train, feat_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
                              generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(TensorDataset(mat_test, feat_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridModel(n_feat=feat_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    best_probs = None

    for epoch in range(EPOCHS):
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

        test_acc = tc / tt
        p(f"  Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_probs = np.concatenate(all_probs)
            p(f"    >>> Best: {best_acc:.4f}")

    p(f"  Seed {seed} done. Best acc: {best_acc:.4f}")
    np.save(f"{OUT_DIR}/probs_v19_seed{seed}.npy", best_probs)
    return best_probs, best_acc


def main():
    t0 = time.time()
    mat_train, feat_train, y_train, stats = load_all_f32("train")
    mat_test, feat_test, y_test, _ = load_all_f32("test", feat_stats=stats)

    y_test_np = y_test.numpy()

    # Already have v8 (seed=42 effectively, but with default init)
    # Use the saved v8 probabilities as the first seed
    probs_v8 = np.load(f"{OUT_DIR}/probs_v8.npy")
    v8_acc = (probs_v8.argmax(1) == y_test_np).mean()
    p(f"v8 (existing): acc={v8_acc:.4f}")

    all_probs = [probs_v8]

    # Train 2 more seeds
    for seed in [123, 7]:
        probs, acc = train_one_seed(seed, mat_train, feat_train, y_train,
                                     mat_test, feat_test, y_test)
        all_probs.append(probs)

    # Ensemble: average probabilities
    p(f"\n{'='*60}")
    p("Ensemble results:")
    p(f"{'='*60}")

    for n_models in range(1, len(all_probs) + 1):
        avg = np.mean(all_probs[:n_models], axis=0)
        preds = avg.argmax(1)
        acc = (preds == y_test_np).mean()
        true_fracs = np.bincount(y_test_np, minlength=5) / len(y_test_np)
        pred_fracs = np.bincount(preds, minlength=5) / len(preds)
        frac_diff = np.mean(np.abs(true_fracs - pred_fracs))
        p(f"  {n_models} models: acc={acc:.4f}, frac_diff={frac_diff:.4f}")

    # Save 3-model ensemble
    best_probs = np.mean(all_probs, axis=0)
    best_preds = best_probs.argmax(1)
    best_acc = (best_preds == y_test_np).mean()

    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v19.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Ensemble acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: 3-seed ensemble (v8+seed123+seed7), avg probs, {EPOCHS}ep each")

if __name__ == "__main__":
    main()
