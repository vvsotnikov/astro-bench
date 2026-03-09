"""Hard Negative Mining: Train on the most confusing examples.

Strategy:
1. Train initial model on all data
2. Identify hadrons that score highest (most gamma-like)
3. Resample training data to over-weight these hard negatives
4. Retrain to improve separation in the critical region
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
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
        return torch.from_numpy(x), int(self.labels[idx])


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


class DNN(nn.Module):
    def __init__(self, input_dim=517, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def compute_survival_at_99(gamma_probs, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(gamma_probs[is_gamma])
    ng = len(sg)
    thr_99 = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
    n_hadron_surviving = (gamma_probs[is_hadron] >= thr_99).sum()
    survival_99 = n_hadron_surviving / is_hadron.sum()
    return survival_99, thr_99


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}\n")

    print("Loading data...")
    raw_train = GammaDataset("train")
    print(f"  Total: {len(raw_train)} events")

    print("Computing normalization...")
    mean, std = compute_stats(raw_train)

    # First pass: train normal model to find hard negatives
    print("\n=== PASS 1: Initial training ===")
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

    model = DNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    train_labels_all = raw_train.labels[train_ds.indices]
    n_gamma_train = (train_labels_all == 0).sum()
    n_hadron_train = (train_labels_all == 1).sum()
    w_gamma = n_hadron_train / (2 * n_gamma_train)
    w_hadron = n_gamma_train / (2 * n_hadron_train)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    print("Training initial model...")
    best_survival = 1.0
    best_epoch = 0

    for epoch in range(40):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Eval
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs[:, 0].cpu().numpy())
                all_labels.append(y.numpy())

        val_probs = np.concatenate(all_probs)
        val_labels = np.concatenate(all_labels)
        val_survival, _ = compute_survival_at_99(val_probs, val_labels)

        if (epoch + 1) % 5 == 0 or val_survival < best_survival:
            print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
                  f"val_surv@99={val_survival:.4f}")

        if val_survival < best_survival:
            best_survival = val_survival
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                      "submissions/haiku-gamma-mar9/model_pass1.pt")

    print(f"Best epoch: {best_epoch}, survival: {best_survival:.4f}\n")

    # Pass 2: Hard negative mining
    print("=== PASS 2: Hard negative mining ===")
    model.load_state_dict(
        torch.load("submissions/haiku-gamma-mar9/model_pass1.pt")
    )
    model.eval()

    # Score all training data
    print("Scoring training data...")
    all_train_probs = []
    all_train_labels = []
    train_full_loader = DataLoader(
        GammaDataset("train", mean=mean, std=std),
        batch_size=8192, shuffle=False, num_workers=8, pin_memory=True
    )
    with torch.no_grad():
        for x, y in train_full_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_train_probs.append(probs[:, 0].cpu().numpy())
            all_train_labels.append(y.numpy())

    train_probs = np.concatenate(all_train_probs)
    train_labels = np.concatenate(all_train_labels)

    # Find hard negatives: hadrons with high gamma probability
    is_hadron = train_labels == 1
    hadron_probs = train_probs[is_hadron]
    hadron_indices = np.where(is_hadron)[0]

    # Sample hard negatives: top 20% by gamma score
    percentile_80 = np.percentile(hadron_probs, 80)
    hard_neg_mask = hadron_probs >= percentile_80
    hard_neg_indices = hadron_indices[hard_neg_mask]
    print(f"Found {hard_neg_mask.sum()} hard negatives (top 20%)")

    # Create sample weights: up-weight hard negatives
    sample_weights = np.ones(len(train_labels), dtype=np.float32)
    sample_weights[hard_neg_indices] *= 5.0  # 5x weight for hard negatives

    # Resample training data
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )
    train_loader_hnm = DataLoader(
        GammaDataset("train", mean=mean, std=std),
        batch_size=4096, sampler=sampler, num_workers=8, pin_memory=True
    )

    print(f"Retraining with hard negative weights...\n")

    model = DNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_survival = 1.0

    for epoch in range(40):
        model.train()
        total_loss = 0
        for x, y in train_loader_hnm:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Eval on validation
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs[:, 0].cpu().numpy())
                all_labels.append(y.numpy())

        val_probs = np.concatenate(all_probs)
        val_labels = np.concatenate(all_labels)
        val_survival, _ = compute_survival_at_99(val_probs, val_labels)

        if (epoch + 1) % 5 == 0 or val_survival < best_survival:
            print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader_hnm):.4f} "
                  f"val_surv@99={val_survival:.4f}")

        if val_survival < best_survival:
            best_survival = val_survival
            torch.save(model.state_dict(),
                      "submissions/haiku-gamma-mar9/model_best_v10.pt")

    # Test
    print("\nEvaluating on test set...")
    model.load_state_dict(
        torch.load("submissions/haiku-gamma-mar9/model_best_v10.pt")
    )
    model.eval()

    test_ds = GammaDataset("test", mean=mean, std=std)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, 0].cpu().numpy())

    test_probs = np.concatenate(all_probs)
    test_survival, test_thr = compute_survival_at_99(test_probs, test_labels)

    print(f"Threshold: {test_thr:.4f}")
    print(f"Test survival @ 99% gamma eff: {test_survival:.4f}")

    np.savez("submissions/haiku-gamma-mar9/predictions_v10.npz",
             gamma_scores=test_probs)


if __name__ == "__main__":
    main()
