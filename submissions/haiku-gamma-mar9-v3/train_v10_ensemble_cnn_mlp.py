"""Ensemble: Average predictions from v3 (Attention CNN) and baseline regression MLP."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim


class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None, is_flat=True):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std
        self.is_flat = is_flat

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)
        feat = self.features[idx].astype(np.float32)

        if self.is_flat:
            mat = mat.flatten()

        x = np.concatenate([mat, feat])
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        label = int(self.labels[idx])
        return torch.from_numpy(x), label


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
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels // 8, 1)
        self.out_proj = nn.Conv2d(channels // 8, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).transpose(1, 2)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w).transpose(1, 2)

        attn = torch.bmm(q, k) * ((c // 8) ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).reshape(b, -1, h, w)
        out = self.out_proj(out)

        return x + out


class AttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mat_conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.mat_attn1 = AttentionBlock(32)

        self.mat_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.mat_attn2 = AttentionBlock(64)

        self.mat_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.mat_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feat_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, mat, feat):
        x = self.mat_conv1(mat)
        x = self.mat_attn1(x)
        x = self.mat_conv2(x)
        x = self.mat_attn2(x)
        x = self.mat_conv3(x)
        x = self.mat_pool(x).view(x.size(0), -1)

        f = self.feat_mlp(feat)

        fused = torch.cat([x, f], dim=1)
        out = self.fusion(fused).squeeze(-1)

        return out


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


def compute_survival_at_75(gamma_scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(gamma_scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (gamma_scores[is_hadron] >= thr).sum()
    return n_hadron_surviving / is_hadron.sum()


device = torch.device("cuda:0")
print(f"Device: {device}\n")

# Load data
print("Loading data...")
raw_train = GammaDataset("train", is_flat=True)
print(f"Computing normalization...")
mean, std = compute_stats(raw_train)

# Test data for both models
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]
is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()


# Split
np.random.seed(42)
torch.manual_seed(42)

n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train

# For flat MLP
print("\n=== Training Baseline Regression MLP ===")
train_ds_flat, val_ds_flat = random_split(
    GammaDataset("train", mean=mean, std=std, is_flat=True),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds_flat, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds_flat, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)

model_mlp = RegDNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model_mlp.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_survival = 1.0
patience = 10

for epoch in range(30):
    model_mlp.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.float().to(device)
        target = (y == 0).float()
        scores = model_mlp(x)
        loss = criterion(scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model_mlp.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                scores = model_mlp(x).cpu().numpy()
                all_scores.append(scores)
                all_labels.append(y.numpy())

        val_scores = np.concatenate(all_scores)
        val_labels = np.concatenate(all_labels)
        val_survival = compute_survival_at_75(val_scores, val_labels)

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 10
            torch.save(model_mlp.state_dict(), "/tmp/model_mlp_ensemble.pt")
        else:
            patience -= 1

model_mlp.load_state_dict(torch.load("/tmp/model_mlp_ensemble.pt"))
model_mlp.eval()

# Get MLP test predictions
test_ds_flat = GammaDataset("test", mean=mean, std=std, is_flat=True)
test_loader_flat = DataLoader(test_ds_flat, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)

mlp_scores = []
with torch.no_grad():
    for x, _ in test_loader_flat:
        x = x.to(device)
        scores = model_mlp(x).cpu().numpy()
        mlp_scores.append(scores)

mlp_test = np.concatenate(mlp_scores)
print(f"MLP survival @ 75%: {compute_survival_75(mlp_test):.4e}")

# Get CNN predictions (from v3)
print("\n=== Loading Attention CNN Predictions ===")
try:
    cnn_npz = np.load("submissions/haiku-gamma-mar9-v3/predictions_v3.npz")
    cnn_test = cnn_npz["gamma_scores"]
    print(f"CNN survival @ 75%: {compute_survival_75(cnn_test):.4e}")
except FileNotFoundError:
    print("ERROR: v3 predictions not found! Need to train v3 first.")
    exit(1)

# Ensemble: average
ensemble_scores = 0.5 * mlp_test + 0.5 * cnn_test
ensemble_surv = compute_survival_75(ensemble_scores)

print(f"\n{'='*60}")
print(f"Ensemble (0.5 * MLP + 0.5 * CNN) @ 75%: {ensemble_surv:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v10.npz",
         gamma_scores=ensemble_scores)

print(f"\n---")
print(f"metric: {ensemble_surv:.4e}")
print(f"description: Ensemble of Attention CNN (v3) + Regression MLP (equal weights)")
