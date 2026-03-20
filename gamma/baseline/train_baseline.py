"""Gamma/hadron baseline: DNN (MLP) on flattened matrices + features.

Binary classifier outputting gamma probability.
Uses load_data and verify modules for data loading and evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from load_data import load_train, load_test
from verify import evaluate


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels, mean=None, std=None):
        self.matrices = matrices
        self.features = features
        self.labels = labels
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


def compute_stats(matrices, features, n_samples=500_000):
    rng = np.random.default_rng(42)
    n = len(matrices)
    indices = rng.choice(n, size=min(n_samples, n), replace=False)
    samples = []
    for idx in indices:
        mat = matrices[idx].flatten().astype(np.float32)
        feat = features[idx].astype(np.float32)
        samples.append(np.concatenate([mat, feat]))
    samples = np.stack(samples)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class DNN(nn.Module):
    def __init__(self, input_dim=517, hidden=512, n_classes=2):
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

    # Load data
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

    # Normalization stats
    print("Computing normalization stats...")
    mean, std = compute_stats(X_train, f_train)

    train_ds = GammaDataset(X_train, f_train, y_train, mean=mean, std=std)
    test_ds = GammaDataset(X_test, f_test, y_test, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)

    model = DNN().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    n_gamma = int((np.array(y_train) == 0).sum())
    n_hadron = int((np.array(y_train) == 1).sum())
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")
    for epoch in range(30):
        model.train()
        total_loss = correct = total = 0
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

        # Evaluate
        model.eval()
        val_loss = 0
        val_total = 0
        all_scores = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                val_total += len(y)
                all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

        epoch_val_loss = val_loss / val_total
        scores = np.concatenate(all_scores)

        print(f"Epoch {epoch+1:2d}/30: train_loss={total_loss/total:.4f} "
              f"train_acc={correct/total:.4f} val_loss={epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_scores = scores
            torch.save(model.state_dict(), "model_baseline.pt")

    # Evaluate through verify.py — this is the ONLY official metric
    evaluate(best_scores, "Baseline: DNN on flattened matrices + features, 30 epochs",
             pred_path="predictions_baseline.npz")

    # Also save predictions for reference
    np.savez("predictions_baseline.npz", gamma_scores=best_scores)


if __name__ == "__main__":
    main()
