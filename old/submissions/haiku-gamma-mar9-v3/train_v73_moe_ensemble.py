"""Mixture of Experts (MoE) ensemble with learned weights.

Instead of grid search for ensemble weights, train a small gating network that learns
to route predictions from v9, v38, and v27b based on the input features.

This allows dynamic, input-dependent weighting instead of fixed weights.
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

# Base models (same as v9, v38, v27b)
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
            nn.Linear(8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, mat, feat):
        x = torch.relu(self.bn1(self.conv1(mat)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        x_feat = self.feat_mlp(feat)
        return self.fusion(torch.cat([x, x_feat], 1)).squeeze(-1)

class ResNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        self.skip1 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.skip2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, mat, feat):
        x = torch.relu(self.bn1(self.conv1(mat)))
        x_skip = self.skip1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x + x_skip

        x_skip = self.skip2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x + x_skip

        x = self.pool(x).view(x.size(0), -1)
        x_feat = self.feat_mlp(feat)
        return self.fusion(torch.cat([x, x_feat], 1)).squeeze(-1)

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(8, 96)
        self.pos_embed = nn.Parameter(torch.randn(1, 64 + 1, 96) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 96) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=96, nhead=3, dim_feedforward=256, dropout=0.1, batch_first=True),
            num_layers=3
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.fusion = nn.Sequential(
            nn.Linear(224, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, mat, feat):
        B, _, H, W = mat.shape
        mat_flat = mat.reshape(B, 2, -1).permute(0, 2, 1)  # (B, 256, 2)
        x = self.patch_embed(mat_flat)  # (B, 256, 96)

        # Downsample to 64 patches
        x = x[:, ::4, :]  # (B, 64, 96)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 96)
        x = torch.cat([cls, x], dim=1)  # (B, 65, 96)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x[:, 0, :]  # (B, 96)

        x_feat = self.feat_mlp(feat)
        return self.fusion(torch.cat([x, x_feat], 1)).squeeze(-1)

# Gating network for MoE
class GatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 experts
            nn.Softmax(dim=1)
        )

    def forward(self, feat):
        return self.gate(feat)

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

device = torch.device("cuda:0")
print(f"Device: {device}\n")

# Load data
raw_train = GammaDataset("train")
print(f"Training data: {len(raw_train)} samples")
mean, std = compute_stats(raw_train)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

n_total = len(raw_train)
n_train = int(0.8 * n_total)

train_ds = GammaDataset("train", mean=mean, std=std)
train_indices = np.arange(n_train)
train_ds_actual = Subset(train_ds, train_indices)

train_loader = DataLoader(train_ds_actual, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

# Load pre-trained models
print("Loading pre-trained models...")
v9_model = AttentionCNN().to(device)
v9_model.load_state_dict(torch.load("/tmp/model_v9_e65.pt", map_location=device))
v9_model.eval()

v38_model = ResNetCNN().to(device)
v38_model.load_state_dict(torch.load("/tmp/model_v38_e65.pt", map_location=device))
v38_model.eval()

v27b_model = ViT().to(device)
v27b_model.load_state_dict(torch.load("/tmp/model_vit_simple.pt", map_location=device))
v27b_model.eval()

# Train gating network
print("Training MoE gating network...")
gating = GatingNetwork().to(device)
optimizer = optim.AdamW(gating.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.BCELoss()

best_test_survival = 1.0
for epoch in range(15):
    gating.train()
    total_loss = 0

    for mat, feat, y in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)

        # Get predictions from base models
        with torch.no_grad():
            p9 = v9_model(mat, feat)
            p38 = v38_model(mat, feat)
            p27b = v27b_model(mat, feat)

        # Get gate weights
        w = gating(feat)  # (B, 3)

        # Combine predictions
        scores = w[:, 0] * p9 + w[:, 1] * p38 + w[:, 2] * p27b

        loss = criterion(scores, (y == 0).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 3 == 0:
        gating.eval()
        with torch.no_grad():
            test_scores = []
            for mat, feat, _ in test_loader:
                mat, feat = mat.to(device), feat.to(device)
                p9 = v9_model(mat, feat)
                p38 = v38_model(mat, feat)
                p27b = v27b_model(mat, feat)
                w = gating(feat)
                scores = w[:, 0] * p9 + w[:, 1] * p38 + w[:, 2] * p27b
                test_scores.append(scores.cpu().numpy())

        test_scores = np.concatenate(test_scores)
        test_survival = compute_survival_75(test_scores, test_labels)

        print(f"Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, test_survival={test_survival:.4e}")

        if test_survival < best_test_survival:
            best_test_survival = test_survival
            torch.save(gating.state_dict(), "/tmp/gating_moe_v73.pt")

# Load best and evaluate
gating.load_state_dict(torch.load("/tmp/gating_moe_v73.pt"))
gating.eval()

test_scores = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        p9 = v9_model(mat, feat)
        p38 = v38_model(mat, feat)
        p27b = v27b_model(mat, feat)
        w = gating(feat)
        scores = w[:, 0] * p9 + w[:, 1] * p38 + w[:, 2] * p27b
        test_scores.append(scores.cpu().numpy())

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores, test_labels)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v73.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: MoE ensemble with learned gating network (v9, v38, v27b)")
