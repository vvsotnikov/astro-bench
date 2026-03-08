"""Published baseline: ELU + BatchNorm DNN from Kuznetsov et al. (JCAP 2024).

Architecture: 514 → 512 → 512 → 512 → 5
Input: flattened 16x16x2 matrices + Ze + Az = 514 dims
This is Model 2 from the paper (~798K params, ~47% accuracy).
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
        # Flatten 16x16x2 = 512, append Ze (idx 1) and Az (idx 2) = 514
        mat = torch.from_numpy(self.matrices[idx].flatten().copy()).float()
        feat = torch.from_numpy(self.features[idx, [1, 2]].copy()).float()  # Ze, Az
        x = torch.cat([mat, feat])
        y = int(self.labels[idx])
        return x, y


class PublishedDNN(nn.Module):
    """ELU + BatchNorm + Dropout, 3 hidden layers of 512."""

    def __init__(self, input_dim=514, n_classes=5):
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

    train_ds = KASCADEDataset("train")
    train_loader = DataLoader(
        train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True
    )

    model = PublishedDNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    criterion = nn.CrossEntropyLoss()

    # Train for 10 epochs
    n_epochs = 10
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

            if batch_idx % 500 == 0:
                print(
                    f"  Epoch {epoch+1}/{n_epochs} batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f} acc={correct/total:.4f}"
                )

        avg_loss = total_loss / total
        avg_acc = correct / total
        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} acc={avg_acc:.4f} lr={lr:.6f}")

    # Predict on test set
    print("\nPredicting on test set...")
    test_ds = KASCADEDataset("test")
    test_loader = DataLoader(
        test_ds, batch_size=8192, shuffle=False, num_workers=4, pin_memory=True
    )

    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(1).cpu().numpy())

    predictions = np.concatenate(all_preds)
    np.savez("submissions/baselines/predictions_published.npz", predictions=predictions)
    print(f"Saved predictions_published.npz ({len(predictions)} predictions)")

    # Also save the model
    torch.save(model.state_dict(), "submissions/baselines/published_dnn.pt")
    print("Saved model weights")


if __name__ == "__main__":
    main()
