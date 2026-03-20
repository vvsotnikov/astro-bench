"""Regression DNN with matrix augmentation (rotations, flips)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim

class GammaDatasetAugment(Dataset):
    def __init__(self, split: str, mean=None, std=None, augment=False):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std
        self.augment = augment
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        feat = self.features[idx].astype(np.float32)

        # Augmentation: random rotation + flip
        if self.augment and self.rng.random() < 0.5:
            k = self.rng.integers(0, 4)  # rotate 0, 90, 180, 270
            mat = np.rot90(mat, k=k)

            if self.rng.random() < 0.5:
                mat = np.fliplr(mat)

        mat_flat = mat.flatten()
        x = np.concatenate([mat_flat, feat])

        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        label = int(self.labels[idx])
        return torch.from_numpy(x), label


def compute_stats(dataset, n_samples=500_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    samples = []
    for idx in indices:
        mat = dataset.matrices[idx].flatten().astype(np.float32)
        feat = dataset.features[idx].astype(np.float32)
        samples.append(np.concatenate([mat, feat]))
    samples = np.stack(samples)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class RegDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(517, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
raw_train = GammaDatasetAugment("train", augment=False)
mean, std = compute_stats(raw_train)

n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train
train_ds, val_ds = random_split(
    GammaDatasetAugment("train", mean=mean, std=std, augment=True),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(2026)
)

train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True,
                         num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False,
                       num_workers=8, pin_memory=True)

test_ds = GammaDatasetAugment("test", mean=mean, std=std, augment=False)
test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False,
                        num_workers=8, pin_memory=True)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

print(f"Training: {len(train_ds)} events")
print(f"Model params: {sum(p.numel() for p in RegDNN().parameters()):,}")

# Train
model = RegDNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

print("\nTraining regression DNN with augmentation...")
best_survival = 1.0
patience = 12

for epoch in range(40):
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
            patience = 12
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
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v28.npz",
         gamma_scores=test_scores)

test_survival = compute_survival_at_75(test_scores, test_labels)
print(f"Test survival @ 75% gamma eff: {test_survival:.4e}")

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Regression DNN with matrix augmentation (rotations, flips)")
