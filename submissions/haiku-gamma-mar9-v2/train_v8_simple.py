"""Simple regression DNN with low dropout."""

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


class SimpleRegDNN(nn.Module):
    """Simple regression DNN."""
    def __init__(self, input_dim=517, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
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


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}\n")

    # Load and prepare data
    print("Loading data...")
    raw_train = GammaDataset("train")
    print(f"  Training: {len(raw_train)} events")

    print("Computing normalization...")
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

    # Train DNN
    model = SimpleRegDNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    print("Training simple regression DNN...")
    best_survival = 1.0
    patience = 20

    for epoch in range(80):
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

        # Eval every other epoch
        if epoch % 2 != 0:
            continue

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

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"val_surv@75={val_survival:.4e} lr={lr:.6f}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 20
            torch.save(model.state_dict(), "submissions/haiku-gamma-mar9-v2/model_best_v8.pt")
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping")
                break

    # Inference on test
    print("\nInference on test set...")
    model.load_state_dict(torch.load("submissions/haiku-gamma-mar9-v2/model_best_v8.pt"))
    model.eval()

    all_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            scores = model(x).cpu().numpy()
            all_scores.append(scores)

    test_scores = np.concatenate(all_scores)

    # Save predictions
    np.savez("submissions/haiku-gamma-mar9-v2/predictions_v8.npz",
             gamma_scores=test_scores)

    # Compute final score
    test_survival = compute_survival_at_75(test_scores, test_labels)
    print(f"Test survival @ 75% gamma eff: {test_survival:.4e}")

    print(f"\n---")
    print(f"metric: {test_survival:.4e}")
    print(f"description: Simple regression (512→512→256→1), 80 epochs, low dropout")


if __name__ == "__main__":
    main()
