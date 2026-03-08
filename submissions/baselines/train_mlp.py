"""Baseline: Dense MLP on flattened matrices + 5 features (517 dims).

Replicates the approach from the published baseline (Kuznetsov et al. JCAP 2024)
but simplified. Uses ELU + BatchNorm.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class KASCADEDataset(Dataset):
    def __init__(self, split: str):
        self.matrices = np.load(f"data/composition_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/composition_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(
            f"data/composition_{split}/labels_composition.npy", mmap_mode="r"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = torch.from_numpy(self.matrices[idx].flatten().copy()).float()
        feat = torch.from_numpy(self.features[idx].copy()).float()
        x = torch.cat([mat, feat])  # 512 + 5 = 517
        y = int(self.labels[idx])
        return x, y


class DenseMLP(nn.Module):
    def __init__(self, input_dim=517, hidden_dims=(512, 256, 128), n_classes=5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ELU(),
                nn.Dropout(0.3),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = KASCADEDataset("train")
    train_loader = DataLoader(
        train_ds, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True
    )

    model = DenseMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    # Train for 3 epochs (quick baseline)
    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

            if batch_idx % 200 == 0:
                print(
                    f"  Epoch {epoch+1}/{n_epochs} batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f} acc={correct/total:.4f}"
                )

        scheduler.step()
        print(
            f"Epoch {epoch+1}: loss={total_loss/total:.4f} "
            f"acc={correct/total:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
        )

    # Predict on test set
    print("\nPredicting on test set...")
    test_ds = KASCADEDataset("test")
    test_loader = DataLoader(
        test_ds, batch_size=8192, shuffle=False, num_workers=4, pin_memory=True
    )

    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(1).cpu().numpy())

    predictions = np.concatenate(all_preds)
    np.savez("submissions/baselines/predictions.npz", predictions=predictions)
    print(f"Saved predictions.npz ({len(predictions)} predictions)")


if __name__ == "__main__":
    main()
