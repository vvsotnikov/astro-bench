"""Convolutional autoencoder for representation learning + engineered features."""

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
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
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


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for detector matrices."""
    def __init__(self):
        super().__init__()
        # Encoder: 2 → 32 → 64 (bottleneck)
        self.enc1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        # Bottleneck: (64, 8, 8) = 4096 dims

        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def encode(self, x):
        x = torch.relu(self.enc1(x))
        x = torch.relu(self.enc2(x))
        return x  # (B, 64, 8, 8)

    def decode(self, z):
        x = torch.relu(self.dec1(z))
        x = torch.sigmoid(self.dec2(x))  # Output in [0,1]
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class HybridRegressor(nn.Module):
    """Combines autoencoder embeddings with engineered features for regression."""
    def __init__(self, ae_model):
        super().__init__()
        self.ae = ae_model
        for param in self.ae.parameters():
            param.requires_grad = False  # Freeze encoder

        # Bottleneck: 64 * 8 * 8 = 4096
        # Features: 8
        self.fusion = nn.Sequential(
            nn.Linear(4096 + 8, 512),
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
        with torch.no_grad():
            z = self.ae.encode(mat)
        z_flat = z.view(z.size(0), -1)
        combined = torch.cat([z_flat, feat], dim=1)
        return self.fusion(combined).squeeze(-1)


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

# Train autoencoder first (unsupervised)
print("\nTraining Convolutional Autoencoder (unsupervised)...")
n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train

train_ds_ae, val_ds_ae = random_split(
    GammaDataset("train", mean=mean, std=std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader_ae = DataLoader(train_ds_ae, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
val_loader_ae = DataLoader(val_ds_ae, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

ae_model = ConvAutoencoder().to(device)
ae_criterion = nn.MSELoss()
ae_optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)
ae_scheduler = optim.lr_scheduler.CosineAnnealingLR(ae_optimizer, T_max=10)

best_ae_loss = float('inf')
for epoch in range(10):
    ae_model.train()
    total_loss = 0
    for mat, feat, _ in train_loader_ae:
        mat = mat.to(device)
        output = ae_model(mat)
        loss = ae_criterion(output, mat)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()

    ae_scheduler.step()

    if epoch % 2 == 0:
        ae_model.eval()
        val_loss = 0
        with torch.no_grad():
            for mat, _, _ in val_loader_ae:
                mat = mat.to(device)
                output = ae_model(mat)
                val_loss += ae_criterion(output, mat).item()

        val_loss /= len(val_loader_ae)
        if val_loss < best_ae_loss:
            best_ae_loss = val_loss
            torch.save(ae_model.state_dict(), "/tmp/ae_model.pt")
        print(f"  Epoch {epoch}, AE loss: {total_loss / len(train_loader_ae):.4f}, val: {val_loss:.4f}")

ae_model.load_state_dict(torch.load("/tmp/ae_model.pt"))
ae_model.eval()

# Now train regressor on frozen encoder + engineered features
print("\nTraining regressor on AE embeddings + features...")
train_ds, val_ds = random_split(
    GammaDataset("train", mean=mean, std=std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

model = HybridRegressor(ae_model).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_survival = 1.0
patience = 10

for epoch in range(20):
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

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 10
            torch.save(model.state_dict(), "/tmp/model_ae.pt")
        else:
            patience -= 1

        print(f"  Epoch {epoch}, survival@75%: {val_survival:.4e}")

model.load_state_dict(torch.load("/tmp/model_ae.pt"))
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

print(f"\nAutoencoder + regressor survival @ 75%: {test_survival:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v25.npz",
         gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Convolutional autoencoder (unsupervised) + frozen encoder + regressor head")
