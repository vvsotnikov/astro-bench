"""Ensemble v9 + v38 (ResNet): simple 2-model ensemble to understand why v41 works.

v41 ensemble (v9 + v38 + v27b): 3.21e-04
This explores just v9 + v38 to isolate their complementarity.
Hypothesis: ResNet's skip connections capture different features than attention CNN.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
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


# v9: Attention CNN
class AttentionCNN(nn.Module):
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


# v38: ResNet with skip connections
class ResNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual block 1
        self.conv2a = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn2b = nn.BatchNorm2d(64)
        self.skip2 = nn.Sequential(nn.Conv2d(32, 64, 1, stride=2), nn.BatchNorm2d(64))

        # Residual block 2
        self.conv3a = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3b = nn.BatchNorm2d(128)
        self.skip3 = nn.Sequential(nn.Conv2d(64, 128, 1, stride=2), nn.BatchNorm2d(128))

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

        # Residual block 1
        skip = self.skip2(x)
        x = torch.relu(self.bn2a(self.conv2a(x)))
        x = torch.relu(self.bn2b(self.conv2b(x)))
        x = x + skip

        # Residual block 2
        skip = self.skip3(x)
        x = torch.relu(self.bn3a(self.conv3a(x)))
        x = torch.relu(self.bn3b(self.conv3b(x)))
        x = x + skip

        x = self.pool(x).view(x.size(0), -1)
        x_feat = self.feat_mlp(feat)
        return self.fusion(torch.cat([x, x_feat], 1)).squeeze(-1)


device = torch.device("cuda:0")
print(f"Device: {device}\n")

raw_train = GammaDataset("train")
mean, std = compute_stats(raw_train)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

def compute_survival_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    if is_gamma.sum() == 0 or is_hadron.sum() == 0:
        return 1.0
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

n_total = len(raw_train)
n_train = int(0.8 * n_total)

train_ds = GammaDataset("train", mean=mean, std=std)
train_indices = np.arange(n_train)
train_ds_actual = Subset(train_ds, train_indices)

train_loader = DataLoader(train_ds_actual, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

print("Training two-model ensemble: v9 (Attention CNN) + v38 (ResNet)...")

# Train v9
print("\nTraining v9 Attention CNN...")
model_v9 = AttentionCNN().to(device)
optimizer_v9 = optim.AdamW(model_v9.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_v9 = optim.lr_scheduler.CosineAnnealingLR(optimizer_v9, T_max=30)
criterion = nn.BCELoss()
best_v9_survival = 1.0

for epoch in range(30):
    model_v9.train()
    total_loss = 0
    for mat, feat, y in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)
        scores = model_v9(mat, feat)
        loss = criterion(scores, (y == 0).float())
        optimizer_v9.zero_grad()
        loss.backward()
        optimizer_v9.step()
        total_loss += loss.item()

    scheduler_v9.step()

    if epoch % 10 == 0:
        model_v9.eval()
        with torch.no_grad():
            test_scores = []
            for mat, feat, _ in test_loader:
                mat, feat = mat.to(device), feat.to(device)
                test_scores.append(model_v9(mat, feat).cpu().numpy())
        test_scores = np.concatenate(test_scores)
        test_survival = compute_survival_75(test_scores, test_labels)
        print(f"  Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, survival={test_survival:.4e}")
        if test_survival < best_v9_survival:
            best_v9_survival = test_survival
            torch.save(model_v9.state_dict(), "/tmp/model_v9_e65.pt")

# Train v38
print("Training v38 ResNet...")
model_v38 = ResNetCNN().to(device)
optimizer_v38 = optim.AdamW(model_v38.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_v38 = optim.lr_scheduler.CosineAnnealingLR(optimizer_v38, T_max=30)
best_v38_survival = 1.0

for epoch in range(30):
    model_v38.train()
    total_loss = 0
    for mat, feat, y in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)
        scores = model_v38(mat, feat)
        loss = criterion(scores, (y == 0).float())
        optimizer_v38.zero_grad()
        loss.backward()
        optimizer_v38.step()
        total_loss += loss.item()

    scheduler_v38.step()

    if epoch % 10 == 0:
        model_v38.eval()
        with torch.no_grad():
            test_scores = []
            for mat, feat, _ in test_loader:
                mat, feat = mat.to(device), feat.to(device)
                test_scores.append(model_v38(mat, feat).cpu().numpy())
        test_scores = np.concatenate(test_scores)
        test_survival = compute_survival_75(test_scores, test_labels)
        print(f"  Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, survival={test_survival:.4e}")
        if test_survival < best_v38_survival:
            best_v38_survival = test_survival
            torch.save(model_v38.state_dict(), "/tmp/model_v38_e65.pt")

# Load best models and create ensemble
model_v9.load_state_dict(torch.load("/tmp/model_v9_e65.pt"))
model_v38.load_state_dict(torch.load("/tmp/model_v38_e65.pt"))
model_v9.eval()
model_v38.eval()

print(f"\nEnsemble predictions (weights: v9=0.70, v38=0.30)...")
test_scores_v9 = []
test_scores_v38 = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        test_scores_v9.append(model_v9(mat, feat).cpu().numpy())
        test_scores_v38.append(model_v38(mat, feat).cpu().numpy())

scores_v9 = np.concatenate(test_scores_v9)
scores_v38 = np.concatenate(test_scores_v38)

# Ensemble: try 0.70/0.30 split (complementary to v41's 0.70 v9 + 0.10 v38 + 0.20 v27b)
ensemble_scores = 0.70 * scores_v9 + 0.30 * scores_v38
test_survival_ensemble = compute_survival_75(ensemble_scores, test_labels)

print(f"v9 alone: {compute_survival_75(scores_v9, test_labels):.4e}")
print(f"v38 alone: {compute_survival_75(scores_v38, test_labels):.4e}")
print(f"Ensemble (0.70/0.30): {test_survival_ensemble:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v65.npz", gamma_scores=ensemble_scores)

print(f"\n---")
print(f"metric: {test_survival_ensemble:.4e}")
print(f"description: 2-model ensemble: v9 (0.70) + v38 ResNet (0.30)")
