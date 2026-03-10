"""Apply Platt scaling calibration to previous best model (v18 seed 42)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from sklearn.calibration import CalibratedClassifierCV


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

# Split: 70% train, 10% calibration, 20% validation
n_samples = len(raw_train)
idx = np.arange(n_samples)
np.random.seed(42)
np.random.shuffle(idx)

n_train = int(0.7 * n_samples)
n_calib = int(0.1 * n_samples)

train_idx = idx[:n_train]
calib_idx = idx[n_train:n_train + n_calib]
val_idx = idx[n_train + n_calib:]

train_ds = Subset(GammaDataset("train", mean=mean, std=std), train_idx.tolist())
calib_ds = Subset(GammaDataset("train", mean=mean, std=std), calib_idx.tolist())
val_ds = Subset(GammaDataset("train", mean=mean, std=std), val_idx.tolist())

train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True)
calib_loader = DataLoader(calib_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)

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


# Train base model
print("Training base regression model...")
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

        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"val_surv@75={val_survival:.4e}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 10
            torch.save(model.state_dict(), "/tmp/model_calib_base.pt")
        else:
            patience -= 1

model.load_state_dict(torch.load("/tmp/model_calib_base.pt"))

# Get calibration set scores
print("Computing calibration set scores...")
model.eval()
calib_scores = []
calib_labels = []
with torch.no_grad():
    for x, y in calib_loader:
        x = x.to(device)
        scores = model(x).cpu().numpy()
        calib_scores.append(scores)
        calib_labels.append(y.numpy())

calib_scores = np.concatenate(calib_scores)
calib_labels = np.concatenate(calib_labels)

# Fit Platt scaling using sklearn's CalibratedClassifierCV with sigmoid method
print("Fitting Platt scaling...")
from sklearn.linear_model import LogisticRegression
platt = LogisticRegression(solver='lbfgs')
# Convert to binary labels for sklearn
calib_y_for_platt = (calib_labels == 0).astype(int)  # 1 for gamma, 0 for hadron
platt.fit(calib_scores.reshape(-1, 1), calib_y_for_platt)

# Get calibrated probabilities for test set
print("Generating test predictions...")
test_scores = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        scores = model(x).cpu().numpy()
        test_scores.append(scores)

test_scores_raw = np.concatenate(test_scores)

# Apply Platt scaling
test_scores_calibrated = platt.predict_proba(test_scores_raw.reshape(-1, 1))[:, 1]

surv_raw = compute_survival_75(test_scores_raw)
surv_calib = compute_survival_75(test_scores_calibrated)

print(f"Test survival @ 75% (raw): {surv_raw:.4e}")
print(f"Test survival @ 75% (Platt): {surv_calib:.4e}")

# Save calibrated predictions
np.savez("submissions/haiku-gamma-mar9-v3/predictions_v7.npz",
         gamma_scores=test_scores_calibrated)

print(f"\n---")
print(f"metric: {surv_calib:.4e}")
print(f"description: Regression DNN + Platt scaling calibration (70/10/20 split)")
