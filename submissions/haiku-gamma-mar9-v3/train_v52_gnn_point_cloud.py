"""Point Cloud / Sparse representation: treat active detector pixels as point cloud (x, y, value)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim


class PointCloudDataset(Dataset):
    """Convert sparse 16x16x2 matrices to point clouds."""
    def __init__(self, split: str, mean=None, std=None, max_points=256):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std
        self.max_points = max_points

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)  # [16, 16, 2]

        # Create point cloud: (x, y, electron_density, muon_density)
        h, w, c = mat.shape
        points = []
        for i in range(h):
            for j in range(w):
                if mat[i, j, 0] > 1e-6 or mat[i, j, 1] > 1e-6:
                    points.append([i/h, j/w, mat[i, j, 0], mat[i, j, 1]])

        if len(points) == 0:
            points = [[0.5, 0.5, 0.0, 0.0]]

        points = np.array(points, dtype=np.float32)

        # Pad or truncate to max_points
        if len(points) < self.max_points:
            pad_len = self.max_points - len(points)
            points = np.vstack([points, np.zeros((pad_len, 4), dtype=np.float32)])
        else:
            points = points[:self.max_points]

        feat = self.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.deg2rad(Ze))
        sin_Ze = np.sin(np.deg2rad(Ze))
        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        if self.mean is not None:
            points_flat = points.flatten()
            points_flat = (points_flat - self.mean[:self.max_points*4]) / (self.std[:self.max_points*4] + 1e-8)
            points = points_flat.reshape(points.shape)
            all_feats = (all_feats - self.mean[self.max_points*4:]) / (self.std[self.max_points*4:] + 1e-8)

        label = int(self.labels[idx])
        return torch.from_numpy(points), torch.from_numpy(all_feats), label


def compute_stats(dataset):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(50_000, len(dataset)), replace=False)

    points_samples = []
    feats_samples = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        points = []
        for i in range(16):
            for j in range(16):
                if mat[i, j, 0] > 1e-6 or mat[i, j, 1] > 1e-6:
                    points.append([i/16, j/16, mat[i, j, 0], mat[i, j, 1]])
        if len(points) == 0:
            points = [[0.5, 0.5, 0.0, 0.0]]
        points = np.array(points, dtype=np.float32)
        if len(points) < 256:
            points = np.vstack([points, np.zeros((256 - len(points), 4), dtype=np.float32)])
        else:
            points = points[:256]
        points_samples.append(points.flatten())

        feat = dataset.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        f = np.array([E, Ze, Az, Ne, Nmu, Ne-Nmu, np.cos(np.deg2rad(Ze)), np.sin(np.deg2rad(Ze))], dtype=np.float32)
        feats_samples.append(f)

    points_samples = np.stack(points_samples)
    feats_samples = np.stack(feats_samples)
    mean = np.concatenate([points_samples.mean(0), feats_samples.mean(0)])
    std = np.concatenate([points_samples.std(0), feats_samples.std(0)])
    std[std == 0] = 1.0
    return mean, std


class PointNetLike(nn.Module):
    """PointNet-style architecture: process points with shared MLP, then aggregate."""
    def __init__(self, max_points=256):
        super().__init__()
        # Point features MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        # Aggregation + classification
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 192),
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
        # points: [B, max_points, 4]
        # Process each point
        x = self.point_mlp(points)  # [B, max_points, 256]

        # Max pooling over points
        x = torch.max(x, dim=1)[0]  # [B, 256]

        x = self.global_mlp(x)  # [B, 128]
        x_feat = self.feat_mlp(feat)
        combined = torch.cat([x, x_feat], dim=1)
        return self.fusion(combined).squeeze(-1)


device = torch.device("cuda:0")
print(f"Device: {device}\n")

raw_train = PointCloudDataset("train")
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

n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train

train_ds, val_ds = random_split(
    PointCloudDataset("train", mean=mean, std=std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)
test_ds = PointCloudDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

print("Training Point Cloud Network...")
model = PointNetLike().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_survival = 1.0
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
            scores = []
            labels = []
            for points, feat, y in val_loader:
                points, feat = points.to(device), feat.to(device)
                scores.append(model(points, feat).cpu().numpy())
                labels.append(y.numpy())
        val_scores = np.concatenate(scores)
        val_labels = np.concatenate(labels)
        is_gamma = val_labels == 0
        is_hadron = val_labels == 1
        sg = np.sort(val_scores[is_gamma])
        ng = len(sg)
        thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
        n_surv = (val_scores[is_hadron] >= thr).sum()
        val_survival = n_surv / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, val_survival={val_survival:.4e}")

        if val_survival < best_survival:
            best_survival = val_survival
            torch.save(model.state_dict(), "/tmp/model_pointcloud.pt")

model.load_state_dict(torch.load("/tmp/model_pointcloud.pt"))
model.eval()

test_scores = []
with torch.no_grad():
    for points, feat, _ in test_loader:
        points, feat = points.to(device), feat.to(device)
        scores = model(points, feat).cpu().numpy()
        test_scores.append(scores)

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v52.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: PointNet-like: sparse detector as point cloud (active pixels), max-pooled aggregation")
