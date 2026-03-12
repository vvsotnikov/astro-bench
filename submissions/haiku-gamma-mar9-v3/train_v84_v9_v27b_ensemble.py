"""Simple 2-model ensemble: v9 + v27b (equal weights).

v41 uses 3 models with optimized weights.
This tests if 2-model unweighted ensemble can beat v9 alone.
v9 + v27b are two different architectures with complementary strengths.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy")[:]
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy")[:]
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

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(8, 96)
        self.pos_embed = nn.Parameter(torch.randn(1, 65, 96) * 0.02)
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
        mat_flat = mat.reshape(B, 2, -1).permute(0, 2, 1)
        x = self.patch_embed(mat_flat)
        x = x[:, ::4, :]  # Downsample to 64 tokens
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x[:, 0, :]
        x_feat = self.feat_mlp(feat)
        return self.fusion(torch.cat([x, x_feat], 1)).squeeze(-1)

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
test_ds = GammaDataset("test")
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

print("Loading v9 and v27b models...")

# Load v9
v9_model = AttentionCNN().to(device)
v9_model.load_state_dict(torch.load("/tmp/model_v9_e65.pt", map_location=device))
v9_model.eval()

# Load v27b
v27b_model = ViT().to(device)
v27b_model.load_state_dict(torch.load("/tmp/model_vit_simple.pt", map_location=device))
v27b_model.eval()

print("Computing ensemble predictions...")

test_scores_v9 = []
test_scores_v27b = []

with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)

        p_v9 = v9_model(mat, feat)
        p_v27b = v27b_model(mat, feat)

        test_scores_v9.append(p_v9.cpu().numpy())
        test_scores_v27b.append(p_v27b.cpu().numpy())

test_scores_v9 = np.concatenate(test_scores_v9)
test_scores_v27b = np.concatenate(test_scores_v27b)

# Equal-weight ensemble
ensemble_scores = (test_scores_v9 + test_scores_v27b) / 2.0

test_survival = compute_survival_75(ensemble_scores, test_labels)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v84.npz", gamma_scores=ensemble_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Simple 2-model ensemble v9 + v27b (equal weights)")
