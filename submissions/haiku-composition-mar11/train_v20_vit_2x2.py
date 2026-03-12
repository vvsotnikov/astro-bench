"""v20: Vision Transformer with 2×2 patches (Phase 3: B3b).

Vision Transformer variant:
- 2×2 patches (64 tokens from 16×16)
- Transformer layers with attention
- 7 features merged in
- log1p preprocessing
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
        mat = np.log1p(mat).transpose(2, 0, 1)
        E, Ze, Az, Ne, Nmu = feat
        eng = np.array([E, np.cos(np.radians(Ze)),
                       np.sin(np.radians(Az)), np.cos(np.radians(Az)),
                       Ne, Nmu, Ne - Nmu], dtype=np.float32)
        return torch.from_numpy(mat), torch.from_numpy(eng), int(self.labels[idx])


class ViT2x2(nn.Module):
    """Vision Transformer with 2×2 patches."""
    def __init__(self, patch_size=2, embed_dim=128, num_heads=4, depth=3, mlp_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (16 // patch_size) ** 2
        patch_dim = 2 * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, batch_first=True, dropout=0.1)
            for _ in range(depth)
        ])

        # Feature fusion
        self.feat_bn = nn.BatchNorm1d(7)
        self.feat_net = nn.Sequential(
            nn.Linear(7, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 64, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    def forward(self, mat, feat):
        B, C, H, W = mat.shape
        P = self.patch_size

        # Extract patches
        patches = mat.reshape(B, C, H // P, P, W // P, P)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // P) * (W // P), C * P * P)

        # Embed patches
        x = self.patch_embed(patches)
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Feature pathway
        f = self.feat_bn(feat)
        f = self.feat_net(f)

        # Fusion
        fused = torch.cat([x, f], dim=1)
        return self.head(fused)


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

    model = ViT2x2().to(DEVICE)
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

    np.savez("submissions/haiku-composition-mar11/predictions_v20.npz",
             predictions=test_preds)

    print(f"\n---")
    print(f"metric: {test_acc:.4f}")
    print(f"description: Vision Transformer 2x2 patches with 7 engineered features")

if __name__ == "__main__":
    main()
