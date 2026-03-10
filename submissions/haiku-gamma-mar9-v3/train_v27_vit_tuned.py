"""Improved Vision Transformer - better tuning for gamma/hadron task."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim


class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1))  # (2, 16, 16)

        feat = self.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.deg2rad(Ze))
        sin_Ze = np.sin(np.deg2rad(Ze))
        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        if self.mean is not None:
            mat_flat = mat.flatten()
            mat_flat = (mat_flat - self.mean[:512]) / (self.std[:512] + 1e-8)
            mat = mat_flat.reshape(mat.shape)
            all_feats = (all_feats - self.mean[512:]) / (self.std[512:] + 1e-8)

        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(all_feats), label


def compute_stats(dataset, n_samples=500_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mat_samples = []
    feat_samples = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1)).flatten()
        mat_samples.append(mat)

        feat = dataset.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.deg2rad(Ze))
        sin_Ze = np.sin(np.deg2rad(Ze))
        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)
        feat_samples.append(all_feats)

    mat_samples = np.stack(mat_samples)
    feat_samples = np.stack(feat_samples)

    mean = np.concatenate([mat_samples.mean(axis=0), feat_samples.mean(axis=0)])
    std = np.concatenate([mat_samples.std(axis=0), feat_samples.std(axis=0)])
    std[std == 0] = 1.0

    return mean, std


class VisionTransformerTuned(nn.Module):
    """Improved ViT with better architecture for gamma/hadron task."""
    def __init__(self, patch_size=2, embed_dim=128, depth=6, heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Smaller patches (2x2 instead of 4x4) = more tokens = more expressiveness
        num_patches = (16 // patch_size) ** 2  # 8x8 = 64 patches
        patch_dim = (patch_size ** 2) * 2  # 2*2*2 = 8

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer encoder with more layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=depth
        )

        # Feature pathway with more capacity
        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
        )

        # Fusion with attention to ViT embeddings
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, mat, feat):
        B = mat.size(0)

        # Patch embedding with smaller patches
        # (B, 2, 16, 16) -> patches of (2, 2)
        mat = mat.reshape(B, 2, 8, self.patch_size, 8, self.patch_size)  # (B, 2, 8, 2, 8, 2)
        mat = mat.permute(0, 2, 4, 1, 3, 5)  # (B, 8, 8, 2, 2, 2)
        mat = mat.reshape(B, 64, self.patch_size**2 * 2)  # (B, 64, 8)
        x = self.patch_embed(mat)  # (B, 64, embed_dim)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 65, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Use CLS token
        x = x[:, 0]  # (B, embed_dim)

        # Feature pathway
        f = self.feat_mlp(feat)

        # Fusion
        fused = torch.cat([x, f], dim=1)
        out = self.fusion(fused).squeeze(-1)

        return out


device = torch.device("cuda:0")
print(f"Device: {device}\n")

# Load and compute stats
print("Loading data...")
raw_train = GammaDataset("train")
print(f"Computing normalization...")
mean, std = compute_stats(raw_train)

test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]
test_is_gamma = test_labels == 0
test_is_hadron = test_labels == 1

def compute_survival_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()


# Train/val split
print("\nPreparing train/val split...")
n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train

train_ds, val_ds = random_split(
    GammaDataset("train", mean=mean, std=std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

# Train ViT
print("Training improved Vision Transformer...")
model = VisionTransformerTuned(patch_size=2, embed_dim=128, depth=6, heads=8).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

best_survival = 1.0
patience = 12

for epoch in range(40):
    model.train()
    total_loss = 0
    for mat, feat, y in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)
        target = (y == 0).float()
        scores = model(mat, feat)
        loss = criterion(scores, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for mat, feat, y in val_loader:
                mat, feat = mat.to(device), feat.to(device)
                scores = model(mat, feat).cpu().numpy()
                all_scores.append(scores)
                all_labels.append(y.numpy())

        val_scores = np.concatenate(all_scores)
        val_labels = np.concatenate(all_labels)
        val_survival = compute_survival_75(val_scores, val_labels)

        print(f"Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, val_survival={val_survival:.4e}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 12
            torch.save(model.state_dict(), "/tmp/model_vit_tuned.pt")
        else:
            patience -= 1

model.load_state_dict(torch.load("/tmp/model_vit_tuned.pt"))
model.eval()

# Get test predictions
test_scores = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        scores = model(mat, feat).cpu().numpy()
        test_scores.append(scores)

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores, test_labels)

print(f"\nViT (tuned) survival @ 75%: {test_survival:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v27.npz",
         gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Vision Transformer (improved) - 2x2 patches, 128D, 6 layers, GELU, grad clip")
