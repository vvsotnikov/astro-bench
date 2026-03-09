"""CNN on 16×16×2 matrices + scalar features."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
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
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        feat = self.features[idx].astype(np.float32)  # (5,)
        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(feat), label


def compute_stats(dataset, n_samples=500_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    feat_samples = []
    mat_samples = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        feat = dataset.features[idx].astype(np.float32)
        mat_samples.append(mat)
        feat_samples.append(feat)

    mats = np.stack(mat_samples)
    feats = np.stack(feat_samples)

    mat_mean = mats.mean(axis=0)
    mat_std = mats.std(axis=0)
    mat_std[mat_std == 0] = 1.0

    feat_mean = feats.mean(axis=0)
    feat_std = feats.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    return (mat_mean, mat_std), (feat_mean, feat_std)


class CNN_Regression(nn.Module):
    """CNN on matrices + MLP on features."""
    def __init__(self):
        super().__init__()
        # CNN: 2 channels input
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # After 2 pooling: 16x16 -> 8x8 -> 4x4, 32 channels
        self.fc_cnn = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # MLP on features
        self.fc_feat = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combined
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, mat, feat):
        # Rearrange from (batch, 16, 16, 2) to (batch, 2, 16, 16)
        mat = mat.permute(0, 3, 1, 2)
        # CNN path
        x = torch.relu(self.bn1(self.conv1(mat)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_cnn(x)

        # Feature path
        f = self.fc_feat(feat)

        # Combine
        combined = torch.cat([x, f], dim=1)
        out = self.fc_combined(combined)
        return out.squeeze(-1)


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
raw_train = GammaDataset("train")
print(f"Computing normalization...")
mat_stats, feat_stats = compute_stats(raw_train)
mat_mean, mat_std = mat_stats
feat_mean, feat_std = feat_stats

# Custom dataset with normalization
class NormGammaDataset(GammaDataset):
    def __init__(self, split: str, mat_stats, feat_stats):
        super().__init__(split)
        self.mat_mean, self.mat_std = mat_stats
        self.feat_mean, self.feat_std = feat_stats

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)
        feat = self.features[idx].astype(np.float32)
        mat = (mat - self.mat_mean) / (self.mat_std + 1e-8)
        feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)
        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(feat), label

# Train/val split
n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train
train_ds, val_ds = random_split(
    NormGammaDataset("train", mat_stats, feat_stats),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(2026)
)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,
                         num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False,
                       num_workers=8, pin_memory=True)

test_ds = NormGammaDataset("test", mat_stats, feat_stats)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False,
                        num_workers=8, pin_memory=True)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

# Train
model = CNN_Regression().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}\n")

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

print("Training CNN...")
best_survival = 1.0
patience = 12

for epoch in range(40):
    model.train()
    total_loss = 0
    for mat, feat, y in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)
        target = (y == 0).float()
        scores = model(mat, feat)
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
            for mat, feat, y in val_loader:
                mat, feat = mat.to(device), feat.to(device)
                scores = model(mat, feat).cpu().numpy()
                all_scores.append(scores)
                all_labels.append(y.numpy())

        val_scores = np.concatenate(all_scores)
        val_labels = np.concatenate(all_labels)
        val_survival = compute_survival_at_75(val_scores, val_labels)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"val_surv@75={val_survival:.4e} lr={lr:.6f}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 12
            torch.save(model.state_dict(), "submissions/haiku-gamma-mar9-v2/model_best_v19.pt")
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping")
                break

# Test
print("\nInference on test set...")
model.load_state_dict(torch.load("submissions/haiku-gamma-mar9-v2/model_best_v19.pt"))
model.eval()

all_scores = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        scores = model(mat, feat).cpu().numpy()
        all_scores.append(scores)

test_scores = np.concatenate(all_scores)
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v19.npz",
         gamma_scores=test_scores)

test_survival = compute_survival_at_75(test_scores, test_labels)
print(f"Test survival @ 75% gamma eff: {test_survival:.4e}")

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: CNN on 16x16x2 matrices + MLP on features")
