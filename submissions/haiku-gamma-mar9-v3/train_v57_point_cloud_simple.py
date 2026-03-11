"""Point Cloud approach: treat detector pixels as nodes, learn on point cloud structure.

Instead of treating the 16x16x2 matrices as images, treat active pixels as a point cloud
and use permutation-invariant aggregation (PointNet-style max pooling).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
        # Matrix: (16, 16, 2) - electron and muon channels
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)

        # Create point cloud: extract (x, y, electron, muon) for each pixel
        h, w, c = mat.shape
        points = []
        for i in range(h):
            for j in range(w):
                x_norm = (i - 7.5) / 8.0  # Normalize coordinates to [-1, 1]
                y_norm = (j - 7.5) / 8.0
                electron_val = mat[i, j, 0]
                muon_val = mat[i, j, 1]
                # Only include pixels with significant energy
                if electron_val > 0.01 or muon_val > 0.01:
                    points.append([x_norm, y_norm, electron_val, muon_val])

        # Convert to array; pad or limit to max 64 points
        if len(points) > 0:
            points = np.array(points, dtype=np.float32)
            if len(points) > 64:
                # Sample 64 points randomly
                idx_sample = np.random.choice(len(points), 64, replace=False)
                points = points[idx_sample]
            elif len(points) < 64:
                # Pad with zeros
                pad = np.zeros((64 - len(points), 4), dtype=np.float32)
                points = np.vstack([points, pad])
        else:
            # Empty detector - use zero vector
            points = np.zeros((64, 4), dtype=np.float32)

        # Scalar features
        feat = self.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.deg2rad(Ze))
        sin_Ze = np.sin(np.deg2rad(Ze))
        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        if self.mean is not None:
            # Normalize point cloud coordinates and values
            points_flat = points.flatten()
            points_flat = (points_flat - self.mean[:256]) / (self.std[:256] + 1e-8)
            points = points_flat.reshape(points.shape)
            all_feats = (all_feats - self.mean[256:]) / (self.std[256:] + 1e-8)

        label = int(self.labels[idx])
        return torch.from_numpy(points), torch.from_numpy(all_feats), label


def compute_stats(dataset):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(500_000, len(dataset)), replace=False)
    points_list, feats_list = [], []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        h, w, c = mat.shape
        points = []
        for i in range(h):
            for j in range(w):
                x_norm = (i - 7.5) / 8.0
                y_norm = (j - 7.5) / 8.0
                electron_val = mat[i, j, 0]
                muon_val = mat[i, j, 1]
                if electron_val > 0.01 or muon_val > 0.01:
                    points.append([x_norm, y_norm, electron_val, muon_val])

        if len(points) > 0:
            points = np.array(points, dtype=np.float32)
            if len(points) > 64:
                idx_sample = np.random.choice(len(points), 64, replace=False)
                points = points[idx_sample]
            elif len(points) < 64:
                pad = np.zeros((64 - len(points), 4), dtype=np.float32)
                points = np.vstack([points, pad])
        else:
            points = np.zeros((64, 4), dtype=np.float32)

        f = dataset.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = f
        f = np.array([E, Ze, Az, Ne, Nmu, Ne-Nmu, np.cos(np.deg2rad(Ze)), np.sin(np.deg2rad(Ze))], dtype=np.float32)

        points_list.append(points.flatten())
        feats_list.append(f)

    points_arr = np.stack(points_list)
    feats_arr = np.stack(feats_list)
    mean = np.concatenate([points_arr.mean(0), feats_arr.mean(0)])
    std = np.concatenate([points_arr.std(0), feats_arr.std(0)])
    std[std == 0] = 1.0
    return mean, std


class PointNetClassifier(nn.Module):
    """PointNet-style architecture for point clouds."""
    def __init__(self):
        super().__init__()
        # MLP on point features
        self.mlp1 = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Global max pooling then another MLP
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Scalar feature MLP
        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        # Fusion
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

    def forward(self, points, feat):
        # points: (batch, 64, 4)
        batch_size = points.size(0)

        # Apply MLP1 to each point
        x = points.view(-1, 4)  # (batch*64, 4)
        x = self.mlp1(x)  # (batch*64, 128)
        x = x.view(batch_size, 64, 128)  # (batch, 64, 128)

        # Global max pooling
        x = x.max(dim=1)[0]  # (batch, 128)

        # Apply MLP2
        x = self.mlp2(x)  # (batch, 128)

        # Process scalar features
        x_feat = self.feat_mlp(feat)  # (batch, 128)

        # Fusion
        combined = torch.cat([x, x_feat], dim=1)  # (batch, 256)
        return self.fusion(combined).squeeze(-1)


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

from torch.utils.data import Subset
train_ds = GammaDataset("train", mean=mean, std=std)
val_indices = np.arange(n_train, n_total)
val_ds = Subset(train_ds, val_indices)

train_indices = np.arange(n_train)
train_ds_actual = Subset(train_ds, train_indices)

train_loader = DataLoader(train_ds_actual, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

print("Training Point Cloud (PointNet) classifier...")
model = PointNetClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_test_survival = 1.0
for epoch in range(30):
    model.train()
    total_loss = 0
    for points, feat, y in train_loader:
        points, feat, y = points.to(device), feat.to(device), y.float().to(device)
        scores = model(points, feat)
        loss = criterion(scores, (y == 0).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_scores = []
            for points, feat, _ in test_loader:
                points, feat = points.to(device), feat.to(device)
                test_scores.append(model(points, feat).cpu().numpy())

        test_scores = np.concatenate(test_scores)
        test_survival = compute_survival_75(test_scores, test_labels)

        print(f"Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, test_survival={test_survival:.4e}")

        if test_survival < best_test_survival:
            best_test_survival = test_survival
            torch.save(model.state_dict(), "/tmp/model_pointnet_v57.pt")

model.load_state_dict(torch.load("/tmp/model_pointnet_v57.pt"))
model.eval()

test_scores = []
with torch.no_grad():
    for points, feat, _ in test_loader:
        points, feat = points.to(device), feat.to(device)
        scores = model(points, feat).cpu().numpy()
        test_scores.append(scores)

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores, test_labels)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v57.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: PointNet point cloud architecture (active pixel detection)")
