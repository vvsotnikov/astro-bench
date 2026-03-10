"""Try v9 architecture with different data split seed."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim


class GammaDataset(Dataset):
    def __init__(self, split: str, mat_mean=None, mat_std=None, feat_mean=None, feat_std=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mat_mean = mat_mean
        self.mat_std = mat_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)
        feat = self.features[idx].astype(np.float32)

        if self.mat_mean is not None:
            mat = (mat - self.mat_mean) / (self.mat_std + 1e-8)

        E = feat[0]
        Ze = feat[1]
        Az = feat[2]
        Ne = feat[3]
        Nmu = feat[4]

        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.radians(Ze * 180 / np.pi))
        sin_Ze = np.sin(np.radians(Ze * 180 / np.pi))

        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        if self.feat_mean is not None:
            all_feats = (all_feats - self.feat_mean) / (self.feat_std + 1e-8)

        mat = np.transpose(mat, (2, 0, 1))

        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(all_feats), label


def compute_stats(dataset, n_samples=100_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mat_samples = []
    feat_samples = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        feat = dataset.features[idx].astype(np.float32)

        E = feat[0]
        Ze = feat[1]
        Az = feat[2]
        Ne = feat[3]
        Nmu = feat[4]

        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.radians(Ze * 180 / np.pi))
        sin_Ze = np.sin(np.radians(Ze * 180 / np.pi))

        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        mat_samples.append(mat)
        feat_samples.append(all_feats)

    mat_samples = np.stack(mat_samples)
    feat_samples = np.stack(feat_samples)

    mat_mean = mat_samples.mean(axis=0)
    mat_std = mat_samples.std(axis=0)
    mat_std[mat_std == 0] = 1.0

    feat_mean = feat_samples.mean(axis=0)
    feat_std = feat_samples.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    return mat_mean, mat_std, feat_mean, feat_std


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


class AttentionCNNWithFeatures(nn.Module):
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
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
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

# Load and normalize
print("Loading data and computing statistics...")
raw_train = GammaDataset("train")
mat_mean, mat_std, feat_mean, feat_std = compute_stats(raw_train)

test_ds = GammaDataset("test", mat_mean, mat_std, feat_mean, feat_std)
test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()


# Train/val split with DIFFERENT seed (seed 2026, year seed)
print("Preparing train/val split with seed=2026...")
np.random.seed(2026)
torch.manual_seed(2026)

n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train
train_ds, val_ds = random_split(
    GammaDataset("train", mat_mean, mat_std, feat_mean, feat_std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(2026)
)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)

# Train
print("Training attention CNN with different data split...")
model = AttentionCNNWithFeatures().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_survival = 1.0
patience = 10

for epoch in range(30):
    model.train()
    total_loss = 0
    for mat, feat, y in train_loader:
        mat, feat = mat.to(device), feat.to(device)
        y = y.float().to(device)
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
        val_survival = compute_survival_at_75(val_scores, val_labels)

        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"val_surv@75={val_survival:.4e}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 10
            torch.save(model.state_dict(), "/tmp/model_different_split.pt")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

# Inference
print("Running inference...")
model.load_state_dict(torch.load("/tmp/model_different_split.pt"))
model.eval()

all_scores = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        scores = model(mat, feat).cpu().numpy()
        all_scores.append(scores)

test_scores = np.concatenate(all_scores)
surv = compute_survival_75(test_scores)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v19.npz",
         gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {surv:.4e}")
print(f"description: Attention CNN + features with different split seed (2026)")
