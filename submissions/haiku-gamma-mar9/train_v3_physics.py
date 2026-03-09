"""Physics-informed approach: use the strongest discriminants directly.

Key observation from explore_gamma.py:
- Gamma Ne-Nmu: median 2.65
- Hadron Ne-Nmu: median 0.96

This ratio is MUCH more discriminative than learned features.
Let's try:
1. Feature engineering: Ne, Nmu, Ne-Nmu, E, Ze
2. Simple model on these 5 features (or fewer)
3. See if we can beat the MLP by leveraging physics
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier


class SimpleDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")[:]
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")[:]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        f = self.features[idx].astype(np.float32)  # E, Ze, Az, Ne, Nmu
        x = np.array([
            f[0],  # E
            f[3],  # Ne
            f[4],  # Nmu
            f[3] - f[4],  # Ne - Nmu (STRONG discriminant)
            f[1],  # Ze
        ], dtype=np.float32)
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(x), int(self.labels[idx])


def compute_stats(dataset):
    """Compute mean/std for normalization."""
    samples = []
    for i in range(len(dataset)):
        _, _ = dataset[i]  # This will fail, let me fix
    # Actually let's do this manually
    features = dataset.features
    x = np.column_stack([
        features[:, 0],
        features[:, 3],
        features[:, 4],
        features[:, 3] - features[:, 4],
        features[:, 1],
    ])
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class SmallDNN(nn.Module):
    """Smaller DNN on engineered features."""
    def __init__(self, input_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
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
    raw_train = SimpleDataset("train")
    print(f"  Total: {len(raw_train)} events")

    print("Computing stats...")
    mean, std = compute_stats(raw_train)
    print(f"  mean: {mean}")
    print(f"  std: {std}\n")

    # Split
    n_train = int(0.8 * len(raw_train))
    n_val = len(raw_train) - n_train
    train_ds, val_ds = random_split(
        SimpleDataset("train", mean=mean, std=std),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(2026)
    )

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False,
                           num_workers=4, pin_memory=True)

    test_ds = SimpleDataset("test", mean=mean, std=std)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

    model = SmallDNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    # Class weights
    train_labels_all = raw_train.labels[train_ds.indices]
    n_gamma_train = (train_labels_all == 0).sum()
    n_hadron_train = (train_labels_all == 1).sum()
    w_gamma = n_hadron_train / (2 * n_gamma_train)
    w_hadron = n_gamma_train / (2 * n_hadron_train)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print("Training...")
    best_survival = 1.0
    patience = 15

    for epoch in range(50):
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

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"val_surv@99={val_survival:.4f} lr={lr:.5f}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 15
            torch.save(model.state_dict(),
                      "submissions/haiku-gamma-mar9/model_best_v3.pt")
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stop at epoch {epoch+1}")
                break

    # Test
    print("\nEvaluating on test set...")
    model.load_state_dict(
        torch.load("submissions/haiku-gamma-mar9/model_best_v3.pt")
    )
    model.eval()

    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, 0].cpu().numpy())

    test_probs = np.concatenate(all_probs)
    test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]
    test_survival, test_thr = compute_survival_at_99(test_probs, test_labels)

    print(f"Threshold: {test_thr:.4f}")
    print(f"Test survival @ 99% gamma eff: {test_survival:.4f}")

    np.savez("submissions/haiku-gamma-mar9/predictions_v3.npz",
             gamma_scores=test_probs)


if __name__ == "__main__":
    main()
