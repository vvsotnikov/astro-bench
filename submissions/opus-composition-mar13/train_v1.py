"""Hybrid CNN+MLP for 5-class composition classification.
CNN processes 16x16x2 matrices, MLP processes engineered features, late fusion."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os

DATA_DIR = "data"
OUT_DIR = "submissions/opus-composition-mar13"
DEVICE = "cuda"
BATCH_SIZE = 2048
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 4

# --- Feature engineering ---
def engineer_features(f):
    """Create engineered features from raw 5 features [E, Ze, Az, Ne, Nmu]."""
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    feats = [
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)),
        np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)),
        np.cos(np.radians(Az)),
        Ne - Nmu,           # log ratio (Ne/Nmu in log space)
        Ne + Nmu,           # log product proxy
        (Ne - Nmu) / (Ne + Nmu + 1e-6),  # normalized ratio
        Ne - E,             # electron excess relative to energy
        Nmu - E,            # muon excess relative to energy
    ]
    return np.stack(feats, axis=1).astype(np.float32)


class KASCADEDataset(Dataset):
    def __init__(self, split="train", subsample=None):
        mat_path = f"{DATA_DIR}/composition_{split}/matrices.npy"
        feat_path = f"{DATA_DIR}/composition_{split}/features.npy"
        label_path = f"{DATA_DIR}/composition_{split}/labels_composition.npy"

        self.matrices = np.load(mat_path, mmap_mode='r')
        raw_feats = np.load(feat_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')

        self.n = len(self.labels)
        if subsample and subsample < self.n:
            rng = np.random.RandomState(42)
            self.indices = rng.choice(self.n, subsample, replace=False)
        else:
            self.indices = None

        # Precompute engineered features
        if self.indices is not None:
            raw = np.array(raw_feats[self.indices], dtype=np.float32)
        else:
            # Process in chunks to avoid OOM
            chunk_size = 500000
            chunks = []
            for i in range(0, self.n, chunk_size):
                end = min(i + chunk_size, self.n)
                chunks.append(np.array(raw_feats[i:end], dtype=np.float32))
            raw = np.concatenate(chunks, axis=0)

        self.features = engineer_features(raw)

        # Compute feature normalization stats from training data
        self.feat_mean = self.features.mean(axis=0)
        self.feat_std = self.features.std(axis=0) + 1e-6
        self.features = (self.features - self.feat_mean) / self.feat_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.indices is not None:
            mat_idx = self.indices[idx]
        else:
            mat_idx = idx

        mat = np.array(self.matrices[mat_idx], dtype=np.float32)
        # log1p transform for sparse matrices
        mat = np.log1p(mat)
        # (H, W, C) -> (C, H, W) for PyTorch
        mat = mat.transpose(2, 0, 1)

        feat = self.features[idx]
        label = int(self.labels[mat_idx])

        return torch.from_numpy(mat), torch.from_numpy(feat), label


class HybridCNNMLP(nn.Module):
    def __init__(self, n_feat=13, n_classes=5):
        super().__init__()

        # CNN branch for 16x16x2 matrices
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )

        # MLP branch for scalar features
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat, feat):
        cnn_out = self.cnn(mat).flatten(1)  # (B, 256)
        feat_out = self.feat_mlp(feat)       # (B, 128)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        return self.head(combined)


def main():
    t0 = time.time()

    print("Loading training data...")
    train_ds = KASCADEDataset("train")
    print(f"  Train samples: {len(train_ds)}, features: {train_ds.features.shape[1]}")

    print("Loading test data...")
    test_ds = KASCADEDataset("test")
    # Normalize test features with train stats
    raw_test_feats_path = f"{DATA_DIR}/composition_test/features.npy"
    raw_test = np.array(np.load(raw_test_feats_path, mmap_mode='r'), dtype=np.float32)
    test_ds.features = (engineer_features(raw_test) - train_ds.feat_mean) / train_ds.feat_std
    print(f"  Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = HybridCNNMLP(n_feat=train_ds.features.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (mat, feat, label) in enumerate(train_loader):
            mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            out = model(mat, feat)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(label)
            correct += (out.argmax(1) == label).sum().item()
            total += len(label)

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}/{len(train_loader)}, "
                      f"loss={loss.item():.4f}, acc={correct/total:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        # Evaluate
        model.eval()
        all_preds = []
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for mat, feat, label in test_loader:
                mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
                out = model(mat, feat)
                preds = out.argmax(1)
                all_preds.append(preds.cpu().numpy())
                test_correct += (preds == label).sum().item()
                test_total += len(label)

        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_best.pt")
            print(f"  New best: {best_acc:.4f}")

    # Save predictions
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"---")
    print(f"metric: {best_acc:.4f}")
    print(f"description: Hybrid CNN+MLP, {EPOCHS} epochs, log1p matrices, 13 engineered features")

if __name__ == "__main__":
    main()
