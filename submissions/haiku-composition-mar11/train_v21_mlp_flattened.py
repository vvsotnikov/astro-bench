"""v21: Pure MLP on flattened matrices + features (Phase 3: B3d variant).

MLP variant with different architecture:
- Flatten 16×16×2 matrix to 512D
- Concatenate with 7 engineered features (519D total)
- Deep MLP: 519→1024→512→256→5
- Heavy dropout to regularize
- OneCycleLR scheduler
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.amp import autocast, GradScaler

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

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
        mat = np.array(self.matrices[idx], dtype=np.float32)
        feat = np.array(self.features[idx], dtype=np.float32)
        mat = np.log1p(mat).flatten()  # Flatten to 512D
        E, Ze, Az, Ne, Nmu = feat
        eng = np.array([E, np.cos(np.radians(Ze)),
                       np.sin(np.radians(Az)), np.cos(np.radians(Az)),
                       Ne, Nmu, Ne - Nmu], dtype=np.float32)
        combined = np.concatenate([mat, eng])  # 519D
        return torch.from_numpy(combined), int(self.labels[idx])


class MLPFlattened(nn.Module):
    """Deep MLP on flattened spatial + engineered features."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(519, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True,
                              persistent_workers=True)

    test_ds = KASCADEDataset(
        'data/composition_test/matrices.npy',
        'data/composition_test/features.npy',
        'data/composition_test/labels_composition.npy',
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)

    model = MLPFlattened().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    scaler = GradScaler('cuda')

    best_acc = 0
    best_model = None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = correct = total = 0

        for i, (x, labels) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast('cuda'):
                out = model(x)
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
        for x, _ in test_loader:
            x = x.to(DEVICE, non_blocking=True)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.append(preds)

    test_preds = np.concatenate(all_preds)
    test_labels = np.array(np.load('data/composition_test/labels_composition.npy', mmap_mode='r'), dtype=int)
    test_acc = (test_preds == test_labels).mean()

    np.savez("submissions/haiku-composition-mar11/predictions_v21.npz",
             predictions=test_preds)

    print(f"\n---")
    print(f"metric: {test_acc:.4f}")
    print(f"description: Deep MLP on flattened (512D) + 7 features (519D total)")

if __name__ == "__main__":
    main()
