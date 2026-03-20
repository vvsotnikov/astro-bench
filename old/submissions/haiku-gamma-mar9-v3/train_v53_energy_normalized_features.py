"""Simple CNN with energy-normalized features (Ne/E, Nmu/E, etc.)."""

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
        mat = self.matrices[idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1))

        feat = self.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat

        # Energy-normalized features
        E = np.maximum(E, 1.0)  # Avoid division by zero
        Ne_over_E = Ne / E
        Nmu_over_E = Nmu / E
        cos_Ze = np.cos(np.deg2rad(Ze))
        sin_Ze = np.sin(np.deg2rad(Ze))
        all_feats = np.array([E, Ze, Az, Ne_over_E, Nmu_over_E, cos_Ze, sin_Ze], dtype=np.float32)

        if self.mean is not None:
            mat_flat = mat.flatten()
            mat_flat = (mat_flat - self.mean[:512]) / (self.std[:512] + 1e-8)
            mat = mat_flat.reshape(mat.shape)
            all_feats = (all_feats - self.mean[512:]) / (self.std[512:] + 1e-8)

        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(all_feats), label


def compute_stats(dataset, n_samples=500_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mat_samples = []
    feat_samples = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1)).flatten()
        mat_samples.append(mat)

        feat = dataset.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        E = np.maximum(E, 1.0)
        all_feats = np.array([E, Ze, Az, Ne/E, Nmu/E, np.cos(np.deg2rad(Ze)), np.sin(np.deg2rad(Ze))], dtype=np.float32)
        feat_samples.append(all_feats)

    mat_samples = np.stack(mat_samples)
    feat_samples = np.stack(feat_samples)

    mean = np.concatenate([mat_samples.mean(axis=0), feat_samples.mean(axis=0)])
    std = np.concatenate([mat_samples.std(axis=0), feat_samples.std(axis=0)])
    std[std == 0] = 1.0

    return mean, std


class SimpleCNN(nn.Module):
    """Simple CNN without broken attention."""
    def __init__(self, n_feats=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feats, 256),
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

    def forward(self, mat, feat):
        x = torch.relu(self.bn1(self.conv1(mat)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)

        x_feat = self.feat_mlp(feat)
        combined = torch.cat([x, x_feat], dim=1)
        return self.fusion(combined).squeeze(-1)


device = torch.device("cuda:0")
print(f"Device: {device}\n")

raw_train = GammaDataset("train")
mean, std = compute_stats(raw_train)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

def compute_survival_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train

train_ds, val_ds = random_split(
    GammaDataset("train", mean=mean, std=std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)
test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

print("Training with energy-normalized features...")
model = SimpleCNN(n_feats=7).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_survival = 1.0
for epoch in range(30):
    model.train()
    total_loss = 0
    for mat, feat, y in train_loader:
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
            scores = [model(m.to(device), f.to(device)).cpu().numpy() for m, f, _ in val_loader]
        val_survival = compute_survival_75(np.concatenate(scores), np.concatenate([y.numpy() for _, _, y in val_loader]))
        print(f"Epoch {epoch}: survival={val_survival:.4e}")
        if val_survival < best_survival:
            best_survival = val_survival
            torch.save(model.state_dict(), "/tmp/model.pt")

model.load_state_dict(torch.load("/tmp/model.pt"))
model.eval()
test_scores = np.concatenate([model(m.to(device), f.to(device)).cpu().numpy() for m, f, _ in test_loader])
test_survival = compute_survival_75(test_scores, test_labels)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v53.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Simple CNN with energy-normalized features (Ne/E, Nmu/E)")
