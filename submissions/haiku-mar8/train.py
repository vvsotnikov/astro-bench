"""v9: CNN+MLP on FULL data, 30 epochs, on-the-fly processing.
Key: keep using mmap, no preloading, just efficient __getitem__."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

        # Engineer features inline
        E, Ze, Az, Ne, Nmu = feat
        eng = np.array([E, np.cos(np.radians(Ze)),
                       np.sin(np.radians(Az)), np.cos(np.radians(Az)),
                       Ne, Nmu, Ne - Nmu], dtype=np.float32)

        return torch.from_numpy(mat), torch.from_numpy(eng), int(self.labels[idx])

class CNNHybridFinal(nn.Module):
    def __init__(self):
        super().__init__()
        # Deeper CNN with BN
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 8
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 4
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 2
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

    model = CNNHybridFinal().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    scaler = GradScaler('cuda')

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = correct = total = 0
        t0 = time.time()

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

        elapsed = time.time() - t0
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_loader):.4f} acc={epoch_acc:.4f} {elapsed:.1f}s")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'submissions/haiku-mar8/model_best.pt')
            print(f"  -> Best ({best_acc:.4f})")

    print(f"\nBest train: {best_acc:.4f}")

    # Inference
    print("Inference...")
    model.load_state_dict(torch.load('submissions/haiku-mar8/model_best.pt', weights_only=True))
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
            # Engineer
            E, Ze, Az, Ne, Nmu = feat[:,0], feat[:,1], feat[:,2], feat[:,3], feat[:,4]
            eng = np.stack([E, np.cos(np.radians(Ze)),
                           np.sin(np.radians(Az)), np.cos(np.radians(Az)),
                           Ne, Nmu, Ne - Nmu], axis=1).astype(np.float32)

            with autocast('cuda'):
                out = model(torch.from_numpy(mat).to(DEVICE),
                           torch.from_numpy(eng).to(DEVICE))
            all_preds.append(out.argmax(1).cpu().numpy())

    predictions = np.concatenate(all_preds)
    print(f"Predictions: {predictions.shape}, dist: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    np.savez('submissions/haiku-mar8/predictions.npz', predictions=predictions.astype(int))
    print("Done")

if __name__ == '__main__':
    main()
