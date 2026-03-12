"""v26: Data augmentation variant (Phase 1: A6 augmentation strategy).

Tests augmentation with 90° rotations (detector is symmetric):
- Haiku-mar8 architecture
- Random 90° rotations during training
- No rotation on validation/test
- Same hyperparameters as haiku-mar8
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import time

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

N_FEAT = 7

class KASCADEDataset(Dataset):
    def __init__(self, matrices_path, features_path, labels_path, augment=False):
        self.matrices = np.load(matrices_path, mmap_mode='r')
        self.features = np.load(features_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        self.n = len(self.labels)
        self.augment = augment
        print(f"  Dataset: {self.n} (augment={augment})")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        mat = np.array(self.matrices[idx], dtype=np.float32)
        feat = np.array(self.features[idx], dtype=np.float32)
        mat = np.log1p(mat)

        # Random 90° rotation if augmenting
        if self.augment:
            k = np.random.randint(0, 4)  # 0, 90, 180, 270 degrees
            mat = np.rot90(mat, k=k, axes=(0, 1))

        mat = mat.transpose(2, 0, 1)

        E, Ze, Az, Ne, Nmu = feat
        eng = np.array([E, np.cos(np.radians(Ze)),
                       np.sin(np.radians(Az)), np.cos(np.radians(Az)),
                       Ne, Nmu, Ne - Nmu], dtype=np.float32)
        return torch.from_numpy(mat), torch.from_numpy(eng), int(self.labels[idx])


class CNNHybridFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.feat_bn = nn.BatchNorm1d(N_FEAT)
        self.feat_net = nn.Sequential(
            nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
        )
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


def main():
    BATCH_SIZE = 4096
    EPOCHS = 30
    LR = 2e-3

    print("Loading data...")
    train_ds = KASCADEDataset(
        'data/composition_train/matrices.npy',
        'data/composition_train/features.npy',
        'data/composition_train/labels_composition.npy',
        augment=True  # Enable augmentation for training
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True,
                              persistent_workers=True)

    test_ds = KASCADEDataset(
        'data/composition_test/matrices.npy',
        'data/composition_test/features.npy',
        'data/composition_test/labels_composition.npy',
        augment=False  # No augmentation for test
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)

    model = CNNHybridFinal().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    scaler = GradScaler('cuda')

    best_acc = 0
    best_model = None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = correct = total = 0

        for i, (mat, feat, labels) in enumerate(train_loader):
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
            if i % 300 == 0:
                print(f"  E{epoch+1} b{i}/{len(train_loader)} loss={loss.item():.4f} acc={correct/total:.4f}")

        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_loader):.4f} acc={epoch_acc:.4f}")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = model.state_dict().copy()
            print(f"  -> Best ({best_acc:.4f})")

    print("\nEvaluating on test set...")
    model.load_state_dict(best_model)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for mat, feat, _ in test_loader:
            mat = mat.to(DEVICE, non_blocking=True)
            feat = feat.to(DEVICE, non_blocking=True)
            out = model(mat, feat)
            preds = out.argmax(1).cpu().numpy()
            all_preds.append(preds)

    test_preds = np.concatenate(all_preds)
    test_labels = np.array(np.load('data/composition_test/labels_composition.npy', mmap_mode='r'), dtype=int)
    test_acc = (test_preds == test_labels).mean()

    np.savez("submissions/haiku-composition-mar11/predictions_v26.npz",
             predictions=test_preds)

    print(f"\n---")
    print(f"metric: {test_acc:.4f}")
    print(f"description: Haiku-mar8 with 90-degree rotation augmentation")

if __name__ == "__main__":
    main()
