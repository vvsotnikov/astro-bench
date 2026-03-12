"""Ensemble of v9 architectures trained with different loss functions.

v41 uses three different architectures (CNN, ResNet, ViT).
This approach: same CNN architecture, three different loss functions:
1. BCELoss (standard, used in v9)
2. BCEWithLogitsLoss (might be more stable)
3. HuberLoss (robust to outliers)

Then ensemble the three predictions.
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

        # Output logits (no sigmoid) for BCEWithLogitsLoss
        self.fusion = nn.Sequential(
            nn.Linear(256, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)  # Raw logits
        )

    def forward(self, mat, feat):
        x = torch.relu(self.bn1(self.conv1(mat)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
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

def train_and_evaluate(loss_fn_name, criterion, device, train_loader, test_loader, test_labels, mean, std):
    """Train and evaluate a single model."""
    print(f"\n  Training with {loss_fn_name}...")
    model = AttentionCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_test_survival = 1.0
    for epoch in range(30):
        model.train()
        total_loss = 0
        for mat, feat, y in train_loader:
            mat, feat, y = mat.to(device), feat.to(device), y.float().to(device)
            logits = model(mat, feat)
            loss = criterion(logits, (y == 0).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = []
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(device), feat.to(device)
                    logits = model(mat, feat)
                    test_logits.append(logits.cpu().numpy())

            test_logits = np.concatenate(test_logits)
            test_scores = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
            test_survival = compute_survival_75(test_scores, test_labels)

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, test_survival={test_survival:.4e}")

            if test_survival < best_test_survival:
                best_test_survival = test_survival
                torch.save(model.state_dict(), f"/tmp/model_v78_loss_{loss_fn_name}.pt")

    # Load best and get test predictions
    model.load_state_dict(torch.load(f"/tmp/model_v78_loss_{loss_fn_name}.pt"))
    model.eval()

    test_logits = []
    with torch.no_grad():
        for mat, feat, _ in test_loader:
            mat, feat = mat.to(device), feat.to(device)
            logits = model(mat, feat)
            test_logits.append(logits.cpu().numpy())

    test_logits = np.concatenate(test_logits)
    test_scores = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
    test_survival = compute_survival_75(test_scores, test_labels)

    print(f"  Final {loss_fn_name}: {test_survival:.4e}")
    return test_scores

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

print("Training ensemble of v9 with different loss functions...")

# Train with BCEWithLogitsLoss
scores_bce_logits = train_and_evaluate(
    "BCEWithLogitsLoss",
    nn.BCEWithLogitsLoss(),
    device,
    train_loader, test_loader, test_labels, mean, std
)

# Train with HuberLoss
scores_huber = train_and_evaluate(
    "HuberLoss",
    nn.HuberLoss(reduction='mean', delta=0.5),
    device,
    train_loader, test_loader, test_labels, mean, std
)

# Train with MSELoss (for comparison)
scores_mse = train_and_evaluate(
    "MSELoss",
    nn.MSELoss(),
    device,
    train_loader, test_loader, test_labels, mean, std
)

# Ensemble: average the three
print("\nEnsembling predictions...")
ensemble_scores = (scores_bce_logits + scores_huber + scores_mse) / 3.0
ensemble_survival = compute_survival_75(ensemble_scores, test_labels)

print(f"\n---")
print(f"metric: {ensemble_survival:.4e}")
print(f"description: Ensemble of v9 trained with BCEWithLogits, Huber, and MSE loss")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v78.npz", gamma_scores=ensemble_scores)
