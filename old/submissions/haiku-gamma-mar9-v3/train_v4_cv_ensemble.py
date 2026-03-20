"""Cross-validation ensemble: train 5-fold CV models and average predictions."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from sklearn.model_selection import KFold


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
        mat = self.matrices[idx].flatten().astype(np.float32)
        feat = self.features[idx].astype(np.float32)
        x = np.concatenate([mat, feat])
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        label = int(self.labels[idx])
        return torch.from_numpy(x), label


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
mean, std = compute_stats(raw_train)

# Load test data
test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()


# Prepare train dataset indices
n_samples = len(raw_train)
indices = np.arange(n_samples)

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

all_test_scores = []
fold_results = []

for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices), 1):
    print(f"\n=== Fold {fold_idx}/5 ===")

    # Create datasets
    train_ds = Subset(GammaDataset("train", mean=mean, std=std), train_indices.tolist())
    val_ds = Subset(GammaDataset("train", mean=mean, std=std), val_indices.tolist())

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True,
                             num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False,
                           num_workers=8, pin_memory=True)

    # Train
    model = RegDNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_survival = 1.0
    patience = 10

    for epoch in range(30):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            target = (y == 0).float()
            scores = model(x)
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
                for x, y in val_loader:
                    x = x.to(device)
                    scores = model(x).cpu().numpy()
                    all_scores.append(scores)
                    all_labels.append(y.numpy())

            val_scores = np.concatenate(all_scores)
            val_labels = np.concatenate(all_labels)
            val_survival = compute_survival_at_75(val_scores, val_labels)

            if val_survival < best_survival:
                best_survival = val_survival
                patience = 10
                torch.save(model.state_dict(), f"/tmp/model_cv_fold{fold_idx}.pt")
            else:
                patience -= 1

    # Inference
    model.load_state_dict(torch.load(f"/tmp/model_cv_fold{fold_idx}.pt"))
    model.eval()

    all_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            scores = model(x).cpu().numpy()
            all_scores.append(scores)

    test_scores = np.concatenate(all_scores)
    surv = compute_survival_75(test_scores)
    fold_results.append(surv)
    all_test_scores.append(test_scores)
    print(f"Fold {fold_idx} test survival @ 75%: {surv:.4e}")

# Average ensemble
ensemble_scores = np.mean(all_test_scores, axis=0)
ensemble_surv = compute_survival_75(ensemble_scores)

print(f"\n{'='*60}")
print(f"5-Fold Cross-Validation Results:")
for i, surv in enumerate(fold_results, 1):
    print(f"  Fold {i}: {surv:.4e}")
print(f"  Mean: {np.mean(fold_results):.4e}")
print(f"  Std:  {np.std(fold_results):.4e}")
print(f"\nEnsemble (averaging CV folds): {ensemble_surv:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v4.npz",
         gamma_scores=ensemble_scores)

print(f"\n---")
print(f"metric: {ensemble_surv:.4e}")
print(f"description: 5-fold CV ensemble of regression models")
