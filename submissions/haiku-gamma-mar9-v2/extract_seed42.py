"""Extract seed 42 predictions from v18 and use as new best."""

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
            nn.Linear(512, 512 // 2),
            nn.BatchNorm1d(512 // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512 // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda:0")

# Load data
raw_train = GammaDataset("train")
mean, std = compute_stats(raw_train)

test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)

# Load seed 42 model
model = RegDNN().to(device)
model.load_state_dict(torch.load("submissions/haiku-gamma-mar9-v2/model_multiseed_42.pt"))
model.eval()

# Inference
all_scores = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        scores = model(x).cpu().numpy()
        all_scores.append(scores)

test_scores = np.concatenate(all_scores)

# Save as new best
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v18_seed42.npz",
         gamma_scores=test_scores)

print(f"Extracted seed 42 predictions: shape {test_scores.shape}")
print(f"Score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")
