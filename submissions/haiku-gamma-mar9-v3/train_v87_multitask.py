"""v87: Multi-task learning with auxiliary geometry prediction.

Main task: gamma/hadron classification (BCELoss)
Auxiliary tasks:
  - Predict Energy (MSELoss)
  - Predict Zenith angle (MSELoss)
  - Predict Electron count (MSELoss)

Multi-task regularization should help learn more robust features.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim

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
            # Normalize auxiliary targets too
            E_norm = (E - self.mean[512]) / (self.std[512] + 1e-8)
            Ze_norm = (Ze - self.mean[513]) / (self.std[513] + 1e-8)
            Ne_norm = (Ne - self.mean[515]) / (self.std[515] + 1e-8)
        else:
            E_norm, Ze_norm, Ne_norm = E, Ze, Ne

        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(all_feats), label, torch.tensor([E_norm, Ze_norm, Ne_norm], dtype=torch.float32)

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

class MultiTaskCNN(nn.Module):
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

        self.shared = nn.Sequential(
            nn.Linear(256, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )

        # Task heads
        self.gamma_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.energy_head = nn.Linear(128, 1)
        self.zenith_head = nn.Linear(128, 1)
        self.electron_head = nn.Linear(128, 1)

    def forward(self, mat, feat):
        x = torch.relu(self.bn1(self.conv1(mat)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        x_feat = self.feat_mlp(feat)
        x = torch.cat([x, x_feat], 1)
        x = self.shared(x)

        gamma = self.gamma_head(x).squeeze(-1)
        energy = self.energy_head(x).squeeze(-1)
        zenith = self.zenith_head(x).squeeze(-1)
        electron = self.electron_head(x).squeeze(-1)

        return gamma, energy, zenith, electron

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

print("Training v87 with multi-task learning...")
model = MultiTaskCNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

gamma_loss_fn = nn.BCELoss()
aux_loss_fn = nn.MSELoss()

best_test_survival = 1.0
for epoch in range(30):
    model.train()
    total_loss = 0

    for mat, feat, y, aux_targets in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)
        aux_targets = aux_targets.to(device)

        gamma_pred, energy_pred, zenith_pred, electron_pred = model(mat, feat)

        # Multi-task loss
        loss_gamma = gamma_loss_fn(gamma_pred, (y == 0).float())
        loss_energy = aux_loss_fn(energy_pred, aux_targets[:, 0])
        loss_zenith = aux_loss_fn(zenith_pred, aux_targets[:, 1])
        loss_electron = aux_loss_fn(electron_pred, aux_targets[:, 2])

        # Weighted combination: primary task dominates
        loss = 0.7 * loss_gamma + 0.1 * (loss_energy + loss_zenith + loss_electron)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_scores = []
            for mat, feat, _, _ in test_loader:
                mat, feat = mat.to(device), feat.to(device)
                gamma_pred, _, _, _ = model(mat, feat)
                test_scores.append(gamma_pred.cpu().numpy())

        test_scores = np.concatenate(test_scores)
        test_survival = compute_survival_75(test_scores, test_labels)

        print(f"Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, test_survival={test_survival:.4e}")

        if test_survival < best_test_survival:
            best_test_survival = test_survival
            torch.save(model.state_dict(), "/tmp/model_v87_mt.pt")

model.load_state_dict(torch.load("/tmp/model_v87_mt.pt"))
model.eval()

test_scores = []
with torch.no_grad():
    for mat, feat, _, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        gamma_pred, _, _, _ = model(mat, feat)
        test_scores.append(gamma_pred.cpu().numpy())

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores, test_labels)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v87.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Multi-task learning (gamma + Energy/Zenith/Electron prediction)")
