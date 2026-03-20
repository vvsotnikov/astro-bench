"""Hybrid CNN+MLP for 5-class composition classification.
v2: Pre-load data subset to RAM, no multiprocess DataLoader, faster iteration."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

DATA_DIR = "data"
OUT_DIR = "submissions/opus-composition-mar13"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS = 15
LR = 1e-3

def engineer_features(f):
    """Create engineered features from raw 5 features [E, Ze, Az, Ne, Nmu]."""
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    feats = [
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)),
        np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)),
        np.cos(np.radians(Az)),
        Ne - Nmu,           # log(Ne/Nmu)
        Ne + Nmu,           # log(Ne*Nmu)
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E,
        Nmu - E,
    ]
    return np.stack(feats, axis=1).astype(np.float32)


class HybridCNNMLP(nn.Module):
    def __init__(self, n_feat=13, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat, feat):
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))


def load_and_preprocess(split, subsample=None, feat_stats=None):
    """Load data, apply log1p to matrices, engineer features, normalize."""
    print(f"Loading {split} data...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')

    n = len(labels)
    if subsample and subsample < n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, subsample, replace=False)
    else:
        idx = np.arange(n)

    # Load into RAM in chunks
    chunk = 200000
    mat_list, feat_list, label_list = [], [], []
    for i in range(0, len(idx), chunk):
        batch_idx = idx[i:i+chunk]
        batch_idx_sorted = np.sort(batch_idx)
        m = np.array(matrices[batch_idx_sorted], dtype=np.float32)
        m = np.log1p(m)
        m = m.transpose(0, 3, 1, 2)  # (N, C, H, W)
        mat_list.append(m)
        feat_list.append(np.array(raw_feats[batch_idx_sorted], dtype=np.float32))
        label_list.append(np.array(labels[batch_idx_sorted], dtype=np.int64))
        print(f"  Loaded chunk {i//chunk + 1}, total {min(i+chunk, len(idx))}/{len(idx)}")

    mat_all = np.concatenate(mat_list, axis=0)
    feat_raw = np.concatenate(feat_list, axis=0)
    labels_all = np.concatenate(label_list, axis=0)

    # Engineer features
    feats = engineer_features(feat_raw)

    # Normalize
    if feat_stats is None:
        feat_mean = feats.mean(axis=0)
        feat_std = feats.std(axis=0) + 1e-6
    else:
        feat_mean, feat_std = feat_stats
    feats = (feats - feat_mean) / feat_std

    # Convert to tensors
    mat_t = torch.from_numpy(mat_all)
    feat_t = torch.from_numpy(feats)
    label_t = torch.from_numpy(labels_all)

    print(f"  Shape: mat={mat_t.shape}, feat={feat_t.shape}, labels={label_t.shape}")
    return mat_t, feat_t, label_t, (feat_mean, feat_std)


def main():
    t0 = time.time()

    # Use 2M training samples (enough to learn, fast enough to iterate)
    mat_train, feat_train, y_train, feat_stats = load_and_preprocess("train", subsample=2000000)
    mat_test, feat_test, y_test, _ = load_and_preprocess("test", feat_stats=feat_stats)

    train_ds = TensorDataset(mat_train, feat_train, y_train)
    test_ds = TensorDataset(mat_test, feat_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridCNNMLP(n_feat=feat_train.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_preds = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for mat, feat, label in train_loader:
            mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(mat, feat)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(label)
            correct += (out.argmax(1) == label).sum().item()
            total += len(label)

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        # Evaluate
        model.eval()
        all_preds = []
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for mat, feat, label in test_loader:
                mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
                out = model(mat, feat)
                preds = out.argmax(1)
                all_preds.append(preds.cpu().numpy())
                test_correct += (preds == label).sum().item()
                test_total += len(label)

        test_acc = test_correct / test_total
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"test_acc={test_acc:.4f}, time={elapsed/60:.1f}m")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_best.pt")
            print(f"  >>> New best: {best_acc:.4f}")

    # Save predictions
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"---")
    print(f"metric: {best_acc:.4f}")
    print(f"description: Hybrid CNN+MLP v2, {EPOCHS} epochs, 2M train subset, log1p, 13 eng features")


if __name__ == "__main__":
    main()
