"""CNN on 16×16×2 matrices ONLY (no scalar features), with cos(zenith) normalization."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim

class MatricesOnlyDataset(Dataset):
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
        feat = self.features[idx].astype(np.float32)  # (5,)

        # cos(zenith) normalization: Ze is feature[1]
        zenith_rad = np.deg2rad(feat[1])
        cos_zenith = np.cos(zenith_rad)
        mat = mat / (cos_zenith + 0.1)  # avoid division by zero at zenith=90

        # Normalize matrices
        if self.mean is not None and self.std is not None:
            mat = (mat - self.mean) / (self.std + 1e-8)

        # Convert to (2, 16, 16) for Conv2d
        mat = np.transpose(mat, (2, 0, 1))

        label = int(self.labels[idx])
        return torch.from_numpy(mat), label


def compute_stats(dataset, n_samples=500_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    samples = []
    features_list = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        feat = dataset.features[idx].astype(np.float32)
        zenith_rad = np.deg2rad(feat[1])
        cos_zenith = np.cos(zenith_rad)
        mat = mat / (cos_zenith + 0.1)
        samples.append(mat)
        features_list.append(feat)

    samples = np.stack(samples)
    # Compute mean/std across spatial dims for each channel
    mean = samples.mean(axis=(0, 2, 3), keepdims=True)  # (1, 2, 1, 1)
    std = samples.std(axis=(0, 2, 3), keepdims=True)
    std[std == 0] = 1.0
    return mean, std


class MatricesOnlyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16→8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8→4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 4→1
        )

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x.squeeze(-1)


def compute_survival_at_75(gamma_scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    if is_gamma.sum() == 0 or is_hadron.sum() == 0:
        return 1.0
    sg = np.sort(gamma_scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (gamma_scores[is_hadron] >= thr).sum()
    return n_hadron_surviving / is_hadron.sum()


device = torch.device("cuda:0")

print("Device:", device)
print("\nLoading data...")
raw_train = MatricesOnlyDataset("train")
mean, std = compute_stats(raw_train)

print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train
train_ds, val_ds = random_split(
    MatricesOnlyDataset("train", mean=mean, std=std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(2026)
)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                         num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                       num_workers=8, pin_memory=True)

test_ds = MatricesOnlyDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                        num_workers=8, pin_memory=True)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

print(f"Training: {len(train_ds)} events")
print(f"Model params: {sum(p.numel() for p in MatricesOnlyCNN().parameters()):,}")

# Train
model = MatricesOnlyCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print("\nTraining CNN on matrices only (with cos(zenith) normalization)...")
best_survival = 1.0
patience = 15

for epoch in range(50):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.float().to(device)
        target = (y == 0).float()
        scores = model(x)
        loss = criterion(scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 2 == 0:
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                scores = model(x).cpu().numpy()
                all_scores.append(scores)
                all_labels.append(y.numpy())

        val_scores = np.concatenate(all_scores)
        val_labels = np.concatenate(all_labels)
        val_survival = compute_survival_at_75(val_scores, val_labels)

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 15
        else:
            patience -= 1
            if patience <= 0:
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} val_surv@75={val_survival:.4e}")

# Test
print("\nTesting...")
model.eval()
all_scores = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        scores = model(x).cpu().numpy()
        all_scores.append(scores)

test_scores = np.concatenate(all_scores)
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v30.npz",
         gamma_scores=test_scores)

test_survival = compute_survival_at_75(test_scores, test_labels)
print(f"Test survival @ 75% gamma eff: {test_survival:.4e}")

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: CNN on 16x16x2 matrices ONLY (no scalar features), cos(zenith) normalized")
