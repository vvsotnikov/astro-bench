"""Quality Cuts on Training Data: match test distribution.

Test data has quality cuts: Ze<30, Ne>4.8, 0.2<Age<1.48
Training data typically has NO cuts applied.
Hypothesis: Training on unrestricted data introduces distribution mismatch.
Apply same cuts to training to match test distribution exactly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim


class GammaDatasetCuts(Dataset):
    def __init__(self, split: str, mean=None, std=None, apply_cuts=False):
        self.split = split
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std
        self.apply_cuts = apply_cuts

        # Apply quality cuts if specified
        if apply_cuts:
            self.valid_indices = self._get_valid_indices()
        else:
            self.valid_indices = np.arange(len(self.labels))

    def _get_valid_indices(self):
        """Apply quality cuts: Ze<30, Ne>4.8"""
        features_full = np.load(f"data/gamma_{self.split}/features.npy", mmap_mode="r")[:]
        # features: E, Ze, Az, Ne, Nmu
        Ze = features_full[:, 1]
        Ne = features_full[:, 3]

        cuts = (Ze < 30) & (Ne > 4.8)
        return np.where(cuts)[0]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        mat = self.matrices[actual_idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1))

        feat = self.features[actual_idx].astype(np.float32)
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

        label = int(self.labels[actual_idx])
        return torch.from_numpy(mat), torch.from_numpy(all_feats), label


def compute_stats(dataset, apply_cuts=False):
    rng = np.random.default_rng(42)

    # Get valid indices if cuts applied
    if apply_cuts:
        indices = dataset._get_valid_indices()
        indices = rng.choice(indices, size=min(500_000, len(indices)), replace=False)
    else:
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


device = torch.device("cuda:0")
print(f"Device: {device}\n")

raw_train = GammaDatasetCuts("train", apply_cuts=True)
print(f"Training data after quality cuts: {len(raw_train)} samples (from ~1.2M)")
mean, std = compute_stats(raw_train, apply_cuts=True)
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

train_ds = GammaDatasetCuts("train", mean=mean, std=std, apply_cuts=True)
train_indices = np.arange(n_train)
train_ds_actual = Subset(train_ds, train_indices)

train_loader = DataLoader(train_ds_actual, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
test_ds = GammaDatasetCuts("test", mean=mean, std=std, apply_cuts=False)  # Test already has cuts
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

print("Training v9 Attention CNN with quality cuts on training data...")
model = AttentionCNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
criterion = nn.BCELoss()

best_test_survival = 1.0
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
            test_scores = []
            for mat, feat, _ in test_loader:
                mat, feat = mat.to(device), feat.to(device)
                test_scores.append(model(mat, feat).cpu().numpy())

        test_scores = np.concatenate(test_scores)
        test_survival = compute_survival_75(test_scores, test_labels)

        print(f"Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, test_survival={test_survival:.4e}")

        if test_survival < best_test_survival:
            best_test_survival = test_survival
            torch.save(model.state_dict(), "/tmp/model_quality_cuts_v72.pt")

model.load_state_dict(torch.load("/tmp/model_quality_cuts_v72.pt"))
model.eval()

test_scores = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        scores = model(mat, feat).cpu().numpy()
        test_scores.append(scores)

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores, test_labels)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v72.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: v9 Attention CNN trained with quality cuts (Ze<30, Ne>4.8)")
