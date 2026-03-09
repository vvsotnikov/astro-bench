"""Quick multi-seed ensemble: train v3 with different random seeds."""

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

# Quick training on 2 seeds
all_test_scores = []
seed_list = [42, 123]

for seed_idx, seed in enumerate(seed_list, 1):
    print(f"\n=== Seed {seed} ===")

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
                torch.save(model.state_dict(), f"submissions/haiku-gamma-mar9-v2/model_multiseed_{seed}.pt")
            else:
                patience -= 1

    # Inference
    model.load_state_dict(torch.load(f"submissions/haiku-gamma-mar9-v2/model_multiseed_{seed}.pt"))
    model.eval()

    all_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            scores = model(x).cpu().numpy()
            all_scores.append(scores)

    test_scores = np.concatenate(all_scores)
    all_test_scores.append(test_scores)

    surv = compute_survival_75(test_scores)
    print(f"Test survival @ 75% gamma eff: {surv:.4e}")

# Average ensemble
ensemble_scores = np.mean(all_test_scores, axis=0)
ensemble_surv = compute_survival_75(ensemble_scores)
print(f"\nEnsemble of {len(seed_list)} seeds: {ensemble_surv:.4e}")

np.savez("submissions/haiku-gamma-mar9-v2/predictions_v18.npz",
         gamma_scores=ensemble_scores)

print(f"\n---")
print(f"metric: {ensemble_surv:.4e}")
print(f"description: Average ensemble of {len(seed_list)} regression models")
