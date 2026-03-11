"""Curriculum Learning: start with easy examples (high-confidence gammas/hadrons), progress to hard ones."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
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
        # Compute difficulty: how "easy" this sample is
        # Gammas: low Nmu = easy, high Nmu = hard
        # Hadrons: high Nmu = easy, low Nmu = hard
        difficulty = abs(Nmu - 3.0) if label == 0 else abs(Nmu - 10.0)
        return torch.from_numpy(mat), torch.from_numpy(all_feats), label, difficulty


def compute_stats(dataset):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(500_000, len(dataset)), replace=False)
    mats, feats = [], []
    for idx in indices:
        m = dataset.matrices[idx].astype(np.float32).transpose(2, 0, 1).flatten()
        f = dataset.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = f
        f = np.array([E, Ze, Az, Ne, Nmu, Ne-Nmu, np.cos(np.deg2rad(Ze)), np.sin(np.deg2rad(Ze))], dtype=np.float32)
        mats.append(m)
        feats.append(f)
    mats, feats = np.stack(mats), np.stack(feats)
    mean = np.concatenate([mats.mean(0), feats.mean(0)])
    std = np.concatenate([mats.std(0), feats.std(0)])
    std[std == 0] = 1.0
    return mean, std


class CurriculumSampler(Sampler):
    """Sample by difficulty: easy first, hard later."""
    def __init__(self, difficulties, epoch, total_epochs):
        self.difficulties = difficulties
        self.epoch = epoch
        self.total_epochs = total_epochs

        # Difficulty threshold: increases over time
        # Epoch 0: only easiest 20%, Epoch 29: all samples
        self.ratio = 0.2 + (0.8 * epoch / max(1, total_epochs - 1))
        self.indices = np.argsort(self.difficulties)
        self.num_samples = int(len(self.difficulties) * self.ratio)
        self.indices = self.indices[:self.num_samples]

    def __iter__(self):
        return iter(np.random.permutation(self.indices))

    def __len__(self):
        return len(self.indices)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, mat, feat):
        x = torch.relu(self.bn1(self.conv1(mat)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        x_feat = self.feat_mlp(feat)
        return self.fusion(torch.cat([x, x_feat], 1)).squeeze(-1)


device = torch.device("cuda:0")
print(f"Device: {device}\n")

raw_train = GammaDataset("train")
mean, std = compute_stats(raw_train)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

def compute_survival_75(scores):
    is_gamma = test_labels == 0
    is_hadron = test_labels == 1
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

# Prepare datasets
from torch.utils.data import random_split, Subset
n_total = len(raw_train)
n_train = int(0.8 * n_total)
n_val = n_total - n_train

train_ds = GammaDataset("train", mean=mean, std=std)
val_ds = GammaDataset("train", mean=mean, std=std)

train_indices = np.arange(n_train)
val_indices = np.arange(n_train, n_total)

# Get difficulty scores for curriculum
difficulties = np.array([train_ds[i][3].item() for i in train_indices])

val_loader = DataLoader(Subset(val_ds, val_indices), batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)
test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

print("Training with Curriculum Learning...")
model = SimpleCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_survival = 1.0
for epoch in range(30):
    # Curriculum: increase difficulty gradually
    sampler = CurriculumSampler(difficulties, epoch, 30)
    train_loader = DataLoader(
        Subset(train_ds, train_indices[list(sampler)]),
        batch_size=2048,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    model.train()
    total_loss = 0
    for mat, feat, y, _ in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)
        scores = model(mat, feat)
        loss = criterion(scores, (y == 0).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            scores = []
            labels = []
            for mat, feat, y, _ in val_loader:
                mat, feat = mat.to(device), feat.to(device)
                scores.append(model(mat, feat).cpu().numpy())
                labels.append(y.numpy())
        val_scores = np.concatenate(scores)
        val_labels = np.concatenate(labels)
        # Compute survival on validation set
        is_gamma = val_labels == 0
        is_hadron = val_labels == 1
        sg = np.sort(val_scores[is_gamma])
        ng = len(sg)
        thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
        n_surv = (val_scores[is_hadron] >= thr).sum()
        val_survival = n_surv / is_hadron.sum() if is_hadron.sum() > 0 else 1.0
        ratio = sampler.ratio if epoch > 0 else 0.2
        print(f"Epoch {epoch} (difficulty ratio={ratio:.2f}): loss={total_loss/len(train_loader):.4f}, val_survival={val_survival:.4e}")

        if val_survival < best_survival:
            best_survival = val_survival
            torch.save(model.state_dict(), "/tmp/model_curriculum.pt")

model.load_state_dict(torch.load("/tmp/model_curriculum.pt"))
model.eval()

test_scores = []
with torch.no_grad():
    for mat, feat, _, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        scores = model(mat, feat).cpu().numpy()
        test_scores.append(scores)

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v54.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Curriculum Learning: start easy (20% samples), end hard (100%)")
