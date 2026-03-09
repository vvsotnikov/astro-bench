"""High-capacity regression DNN with aggressive regularization."""

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


class HighCapDNN(nn.Module):
    """High-capacity DNN with aggressive regularization."""
    def __init__(self, input_dim=517):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
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

# Load and prepare data
print("Loading data...")
raw_train = GammaDataset("train")
print(f"Computing normalization...")
mean, std = compute_stats(raw_train)

# Train/val split
n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train
train_ds, val_ds = random_split(
    GammaDataset("train", mean=mean, std=std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(2026)
)

train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True,
                         num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False,
                       num_workers=8, pin_memory=True)

test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False,
                        num_workers=8, pin_memory=True)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

# Train
model = HighCapDNN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}\n")

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("Training high-capacity DNN...")
best_survival = 1.0
no_improve = 0

for epoch in range(100):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.float().to(device)
        target = (y == 0).float()
        scores = model(x)
        loss = criterion(scores, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Eval
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

    scheduler.step(val_survival)
    lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch+1:3d}: loss={total_loss/len(train_loader):.4f} "
          f"val_surv@75={val_survival:.4e} lr={lr:.6f}")

    if val_survival < best_survival:
        best_survival = val_survival
        no_improve = 0
        torch.save(model.state_dict(), "submissions/haiku-gamma-mar9-v2/model_best_v22.pt")
    else:
        no_improve += 1
        if no_improve >= 20:
            print(f"No improvement for 20 epochs, stopping")
            break

# Test
print("\nInference on test set...")
model.load_state_dict(torch.load("submissions/haiku-gamma-mar9-v2/model_best_v22.pt"))
model.eval()

all_scores = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        scores = model(x).cpu().numpy()
        all_scores.append(scores)

test_scores = np.concatenate(all_scores)
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v22.npz",
         gamma_scores=test_scores)

test_survival = compute_survival_at_75(test_scores, test_labels)
print(f"Test survival @ 75% gamma eff: {test_survival:.4e}")

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: High-capacity DNN (2048→2048→1024→512→256→1) with aggressive regularization")
