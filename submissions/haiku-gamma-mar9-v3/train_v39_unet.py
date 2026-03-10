"""U-Net style architecture with encoder-decoder for spatial feature extraction."""

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
        mat = np.transpose(mat, (2, 0, 1))

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


class UNet(nn.Module):
    """Simplified U-Net for feature extraction."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_bottle = nn.BatchNorm2d(128)

        # Decoder with skip connections
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 due to skip
        self.bn_dec1 = nn.BatchNorm2d(64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 64 due to skip
        self.bn_dec2 = nn.BatchNorm2d(32)

        # Feature pathway
        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(32 * 16 * 16 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, mat, feat):
        # Encoder
        e1 = torch.relu(self.bn1(self.enc1(mat)))
        p1 = self.pool1(e1)

        e2 = torch.relu(self.bn2(self.enc2(p1)))
        p2 = self.pool2(e2)

        # Bottleneck
        b = torch.relu(self.bn_bottle(self.bottleneck(p2)))

        # Decoder with skip
        d1 = self.upconv1(b)
        d1 = torch.cat([d1, e2], dim=1)  # Skip connection
        d1 = torch.relu(self.bn_dec1(self.dec1(d1)))

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection
        d2 = torch.relu(self.bn_dec2(self.dec2(d2)))

        # Flatten and fuse
        x_spatial = d2.view(d2.size(0), -1)

        x_feat = self.feat_mlp(feat)
        combined = torch.cat([x_spatial, x_feat], dim=1)
        return self.fusion(combined).squeeze(-1)


device = torch.device("cuda:0")
print(f"Device: {device}\n")

print("Loading data...")
raw_train = GammaDataset("train")
print(f"Computing normalization...")
mean, std = compute_stats(raw_train)

test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

def compute_survival_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

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

print("Training U-Net...")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_survival = 1.0
patience = 10

for epoch in range(30):
    model.train()
    total_loss = 0
    for mat, feat, y in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)
        target = (y == 0).float()
        scores = model(mat, feat)
        loss = criterion(scores, target)
        optimizer.zero_grad()
        loss.backward()
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
            patience = 10
            torch.save(model.state_dict(), "/tmp/model_unet.pt")
        else:
            patience -= 1

model.load_state_dict(torch.load("/tmp/model_unet.pt"))
model.eval()

test_scores = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        scores = model(mat, feat).cpu().numpy()
        test_scores.append(scores)

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores, test_labels)

print(f"\nU-Net survival @ 75%: {test_survival:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v39.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: U-Net encoder-decoder with skip connections + engineered features")
