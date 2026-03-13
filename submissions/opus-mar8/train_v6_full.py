"""v6: CNN+Attn+MLP using ALL training data via mmap Dataset.
Uses label smoothing to reduce overconfident misclassification.
num_workers=0, AMP."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS = 15  # fewer epochs since more data per epoch
LR = 1e-3
LABEL_SMOOTH = 0.1

def p(msg):
    print(msg, flush=True)

def engineer_features_single(f):
    """Engineer features for a single sample or batch."""
    E, Ze, Az, Ne, Nmu = f[..., 0], f[..., 1], f[..., 2], f[..., 3], f[..., 4]
    feats = np.stack([
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)),
        np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)),
        np.cos(np.radians(Az)),
        Ne - Nmu,
        Ne + Nmu,
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E,
        Nmu - E,
    ], axis=-1)
    return feats.astype(np.float32)


class MmapDataset(Dataset):
    """Dataset that reads from mmap files on-the-fly."""
    def __init__(self, split, feat_stats=None):
        self.matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
        raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
        self.labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
        self.n = len(self.labels)

        # Precompute engineered features in chunks (small enough for RAM)
        p(f"  Engineering features for {split} ({self.n} samples)...")
        chunk = 500000
        feat_chunks = []
        for i in range(0, self.n, chunk):
            end = min(i + chunk, self.n)
            f = np.array(raw_feats[i:end], dtype=np.float32)
            feat_chunks.append(engineer_features_single(f))
        self.features = np.concatenate(feat_chunks, axis=0)

        if feat_stats is None:
            self.feat_mean = self.features.mean(0)
            self.feat_std = self.features.std(0) + 1e-6
        else:
            self.feat_mean, self.feat_std = feat_stats
        self.features = (self.features - self.feat_mean) / self.feat_std

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        mat = np.array(self.matrices[idx], dtype=np.float32)
        mat = np.log1p(mat).transpose(2, 0, 1)  # (C, H, W)
        feat = self.features[idx]
        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(feat), label


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


def main():
    t0 = time.time()

    p("Setting up datasets...")
    train_ds = MmapDataset("train")
    test_ds = MmapDataset("test", feat_stats=(train_ds.feat_mean, train_ds.feat_std))
    p(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = HybridModel(n_feat=train_ds.features.shape[1]).to(DEVICE)
    p(f"Params: {sum(pp.numel() for pp in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    best_preds = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for bi, (mat_b, feat_b, label_b) in enumerate(train_loader):
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
            if bi % 500 == 0:
                p(f"  Ep {epoch+1} batch {bi}/{len(train_loader)} acc={correct/total:.4f}")
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_preds, all_probs = [], []
        tc, tt = 0, 0
        with torch.no_grad():
            for mat_b, feat_b, label_b in test_loader:
                mat_b, feat_b, label_b = mat_b.to(DEVICE), feat_b.to(DEVICE), label_b.to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mat_b, feat_b)
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
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v6.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v6.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: CNN+Attn+MLP AMP, {EPOCHS}ep, ALL 5.5M train, label_smooth={LABEL_SMOOTH}")

if __name__ == "__main__":
    main()
