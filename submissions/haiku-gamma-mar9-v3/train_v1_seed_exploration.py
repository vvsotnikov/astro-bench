"""Systematic seed exploration: test 20 seeds to find good initializations."""

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


# Test 20 seeds
results = []
seed_list = [42, 123, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
             37, 73, 101, 179, 251, 353, 457, 599, 701, 911]

print(f"Testing {len(seed_list)} seeds...\n")

for seed_idx, seed in enumerate(seed_list, 1):
    print(f"[{seed_idx}/{len(seed_list)}] Seed {seed}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_train = int(0.8 * len(raw_train))
    n_val = len(raw_train) - n_train
    train_ds, val_ds = random_split(
        GammaDataset("train", mean=mean, std=std),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

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
    best_epoch = 0
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
                best_epoch = epoch
                patience = 10
                torch.save(model.state_dict(), f"/tmp/model_seed_{seed}.pt")
            else:
                patience -= 1

    # Inference
    model.load_state_dict(torch.load(f"/tmp/model_seed_{seed}.pt"))
    model.eval()

    all_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            scores = model(x).cpu().numpy()
            all_scores.append(scores)

    test_scores = np.concatenate(all_scores)
    surv = compute_survival_75(test_scores)
    results.append((seed, surv))
    print(f"  → Test survival @ 75%: {surv:.4e}")

print(f"\n{'='*60}")
print("Summary (sorted by metric):")
results_sorted = sorted(results, key=lambda x: x[1])
for seed, surv in results_sorted:
    marker = " ← BEST" if surv == results_sorted[0][1] else ""
    print(f"  Seed {seed:5d}: {surv:.4e}{marker}")

best_seed, best_metric = results_sorted[0]
print(f"\n---")
print(f"metric: {best_metric:.4e}")
print(f"description: Seed exploration ({len(seed_list)} seeds), best seed={best_seed}")
