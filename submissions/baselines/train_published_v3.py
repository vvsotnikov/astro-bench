"""Published baseline v3: with input normalization (critical missing piece)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class KASCADEDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        matrices = np.load(f"data/composition_{split}/matrices.npy", mmap_mode="r")
        features = np.load(f"data/composition_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(
            f"data/composition_{split}/labels_composition.npy", mmap_mode="r"
        )
        # Pre-compute flattened + concatenated input
        # We can't pre-compute everything (too much RAM), so we store raw and normalize on the fly
        self.matrices = matrices
        self.features = features
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].flatten().astype(np.float32)
        feat = self.features[idx].astype(np.float32)
        x = np.concatenate([mat, feat])  # 517
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return torch.from_numpy(x), int(self.labels[idx])


def compute_stats(dataset, n_samples=500_000):
    """Compute mean/std from a subsample of training data."""
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


class PublishedDNN(nn.Module):
    def __init__(self, input_dim=517, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Compute normalization stats from training data
    print("Computing normalization stats...")
    raw_train = KASCADEDataset("train")
    mean, std = compute_stats(raw_train)
    print(f"  mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  std range:  [{std.min():.2f}, {std.max():.2f}]")

    train_ds = KASCADEDataset("train", mean=mean, std=std)
    train_loader = DataLoader(
        train_ds, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True
    )

    model = PublishedDNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 15
    best_acc = 0
    for epoch in range(n_epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        scheduler.step()
        acc = correct / total
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}/{n_epochs}: loss={total_loss/total:.4f} acc={acc:.4f} lr={lr:.6f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "submissions/baselines/best_v3.pt")

    print(f"\nBest train acc: {best_acc:.4f}")
    model.load_state_dict(torch.load("submissions/baselines/best_v3.pt", weights_only=True))

    test_ds = KASCADEDataset("test", mean=mean, std=std)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            logits = model(x.to(device))
            all_preds.append(logits.argmax(1).cpu().numpy())

    predictions = np.concatenate(all_preds)
    np.savez("submissions/baselines/predictions_v3.npz", predictions=predictions)
    print(f"Saved ({len(predictions)} predictions)")


if __name__ == "__main__":
    main()
