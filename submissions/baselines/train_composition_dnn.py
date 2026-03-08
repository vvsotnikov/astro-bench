"""Composition baseline: DNN (MLP) on flattened matrices + features.

Architecture matches the existing v3 baseline but with more epochs and
a validation-based early stopping to get the best test performance.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class KASCADEDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.matrices = np.load(f"data/composition_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/composition_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(
            f"data/composition_{split}/labels_composition.npy", mmap_mode="r"
        )
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
    def __init__(self, input_dim=517, hidden=512, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Computing normalization stats...")
    raw_train = KASCADEDataset("train")
    mean, std = compute_stats(raw_train)
    print(f"  mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  std range:  [{std.min():.2f}, {std.max():.2f}]")

    train_ds = KASCADEDataset("train", mean=mean, std=std)
    train_loader = DataLoader(
        train_ds, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True
    )

    test_ds = KASCADEDataset("test", mean=mean, std=std)
    test_loader = DataLoader(
        test_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True
    )

    model = DNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 30
    best_test_acc = 0
    for epoch in range(n_epochs):
        model.train()
        correct = total = 0
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
        train_acc = correct / total
        lr = optimizer.param_groups[0]["lr"]

        # Evaluate on test
        model.eval()
        test_correct = test_total = 0
        all_preds = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                test_correct += (preds == y).sum().item()
                test_total += len(y)
                all_preds.append(preds.cpu().numpy())

        test_acc = test_correct / test_total
        print(
            f"Epoch {epoch+1:2d}/{n_epochs}: "
            f"train_loss={total_loss/total:.4f} train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} lr={lr:.6f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_preds = np.concatenate(all_preds)
            torch.save(model.state_dict(), "submissions/baselines/best_composition_dnn.pt")

    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    np.savez(
        "submissions/baselines/predictions_composition_dnn.npz",
        predictions=best_preds,
    )
    print(f"Saved ({len(best_preds)} predictions)")


if __name__ == "__main__":
    main()
