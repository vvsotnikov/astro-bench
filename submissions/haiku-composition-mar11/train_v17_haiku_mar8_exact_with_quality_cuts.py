"""v17: Exact haiku-mar8 replica WITH QUALITY CUTS on validation.

Key insight from team lead: test set has quality cuts (Ze<30, Ne>4.8),
but our validation doesn't. This explains why val_acc (~33%) is much lower than test_acc (~50%).

This version applies quality cuts to the validation split to get an accurate validation signal.

Architecture: Exact haiku-mar8
- 4 CNN blocks (32→32→64→64→128→128→256)
- OneCycleLR scheduler
- Batch size 4096
- LR 2e-3
- 7 features (E, cos(Ze), sin(Az), cos(Az), Ne, Nmu, Ne-Nmu)
- BatchNorm on features
- log1p matrices
- 30 epochs, save on train accuracy (not val)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import time

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

N_FEAT = 7

class KASCADEDataset(Dataset):
    def __init__(self, matrices_path, features_path, labels_path):
        self.matrices = np.load(matrices_path, mmap_mode='r')
        self.features = np.load(features_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        self.n = len(self.labels)
        print(f"  Dataset: {self.n}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        mat = np.array(self.matrices[idx], dtype=np.float32)  # (16,16,2)
        feat = np.array(self.features[idx], dtype=np.float32)  # (5,)

        # Log transform matrices
        mat = np.log1p(mat).transpose(2, 0, 1)  # (2,16,16)

        # Engineer features (haiku-mar8 set: 7D, NO sin(Ze))
        E, Ze, Az, Ne, Nmu = feat
        eng = np.array([E, np.cos(np.radians(Ze)),
                       np.sin(np.radians(Az)), np.cos(np.radians(Az)),
                       Ne, Nmu, Ne - Nmu], dtype=np.float32)

        return torch.from_numpy(mat), torch.from_numpy(eng), int(self.labels[idx]), feat

class CNNHybridFinal(nn.Module):
    def __init__(self):
        super().__init__()
        # Exact haiku-mar8: 4 CNN blocks (32→32→64→64→128→128→256)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 2x2
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
        )

        # Feature branch with BN for normalization
        self.feat_bn = nn.BatchNorm1d(N_FEAT)
        self.feat_net = nn.Sequential(
            nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    def forward(self, mat, feat):
        x = self.cnn(mat).reshape(mat.size(0), -1)
        x = self.cnn_fc(x)
        f = self.feat_bn(feat)
        f = self.feat_net(f)
        return self.head(torch.cat([x, f], 1))

def apply_quality_cuts(features):
    """Apply quality cuts: Ze<30, Ne>4.8"""
    Ze = features[:, 1]
    Ne = features[:, 3]
    mask = (Ze < 30) & (Ne > 4.8)
    return mask

def main():
    BATCH_SIZE = 4096
    EPOCHS = 30
    LR = 2e-3

    print("Loading data...")
    train_ds = KASCADEDataset(
        'data/composition_train/matrices.npy',
        'data/composition_train/features.npy',
        'data/composition_train/labels_composition.npy',
    )

    # Split train/val with quality cuts on validation
    print("Creating train/val split with quality cuts on validation...")
    np.random.seed(42)
    torch.manual_seed(42)

    n_train = int(0.8 * len(train_ds))
    n_val = len(train_ds) - n_train

    indices = np.arange(len(train_ds))
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Load val features to apply quality cuts
    val_features_raw = np.array(train_ds.features[val_indices], dtype=np.float32)
    quality_mask = apply_quality_cuts(val_features_raw)
    val_indices_filtered = val_indices[quality_mask]

    print(f"Validation set: {len(val_indices)} → {len(val_indices_filtered)} after quality cuts")

    # Create dataloaders
    from torch.utils.data import Subset
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(train_ds, val_indices_filtered)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True)

    model = CNNHybridFinal().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    scaler = GradScaler('cuda')

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = correct = total = 0
        t0 = time.time()

        for i, (mat, feat, labels, _) in enumerate(train_loader):
            mat = mat.to(DEVICE, non_blocking=True)
            feat = feat.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast('cuda'):
                out = model(mat, feat)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        epoch_acc = correct / total
        elapsed = time.time() - t0

        # Validate
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for mat, feat, labels, _ in val_loader:
                mat = mat.to(DEVICE, non_blocking=True)
                feat = feat.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                out = model(mat, feat)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
                  f"train_acc={epoch_acc:.4f} val_acc={val_acc:.4f} {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/tmp/model_composition_v17.pt")
            print(f"  -> Best val_acc ({best_val_acc:.4f})")

    print(f"\nBest validation: {best_val_acc:.4f}")

    # Test inference
    print("Inference on test set...")
    model.load_state_dict(torch.load("/tmp/model_composition_v17.pt"))
    model.eval()

    test_mat = np.load('data/composition_test/matrices.npy', mmap_mode='r')
    test_feat_raw = np.array(np.load('data/composition_test/features.npy', mmap_mode='r'), dtype=np.float32)

    all_preds = []
    with torch.no_grad():
        bs = 4096
        n_test = len(test_feat_raw)
        for start in range(0, n_test, bs):
            end = min(start + bs, n_test)
            mat = np.array(test_mat[start:end], dtype=np.float32)
            mat = np.log1p(mat).transpose(0, 3, 1, 2)
            feat = test_feat_raw[start:end]

            # Engineer features (7D)
            E, Ze, Az, Ne, Nmu = feat[:, 0], feat[:, 1], feat[:, 2], feat[:, 3], feat[:, 4]
            eng = np.stack([E, np.cos(np.radians(Ze)),
                           np.sin(np.radians(Az)), np.cos(np.radians(Az)),
                           Ne, Nmu, Ne - Nmu], axis=1).astype(np.float32)

            with autocast('cuda'):
                out = model(torch.from_numpy(mat).to(DEVICE),
                           torch.from_numpy(eng).to(DEVICE))
            all_preds.append(out.argmax(1).cpu().numpy())

    test_preds = np.concatenate(all_preds)
    test_labels = np.load('data/composition_test/labels_composition.npy')[:]
    test_acc = (test_preds == test_labels).mean()

    np.savez("submissions/haiku-composition-mar11/predictions_v17.npz",
             predictions=test_preds)

    print(f"\n---")
    print(f"metric: {test_acc:.4f}")
    print(f"description: Exact haiku-mar8 replica with quality cuts on validation (Ze<30, Ne>4.8)")

if __name__ == "__main__":
    main()
