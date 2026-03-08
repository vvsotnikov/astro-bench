"""Published baseline v2: ELU + BatchNorm DNN, all 5 features, proper training."""

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
        # Flatten 16x16x2 = 512, append all 5 features = 517
        mat = torch.from_numpy(self.matrices[idx].flatten().copy()).float()
        feat = torch.from_numpy(self.features[idx].copy()).float()
        x = torch.cat([mat, feat])
        y = int(self.labels[idx])
        return x, y


class PublishedDNN(nn.Module):
    def __init__(self, input_dim=517, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = KASCADEDataset("train")
    train_loader = DataLoader(
        train_ds, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True
    )

    model = PublishedDNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 20
    best_acc = 0
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

        scheduler.step()
        avg_loss = total_loss / total
        avg_acc = correct / total
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f} acc={avg_acc:.4f} lr={lr:.6f}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), "submissions/baselines/best_published_v2.pt")

    # Load best model and predict
    print(f"\nBest train acc: {best_acc:.4f}")
    model.load_state_dict(torch.load("submissions/baselines/best_published_v2.pt", weights_only=True))

    print("Predicting on test set...")
    test_ds = KASCADEDataset("test")
    test_loader = DataLoader(
        test_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True
    )

    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(1).cpu().numpy())

    predictions = np.concatenate(all_preds)
    np.savez("submissions/baselines/predictions_published_v2.npz", predictions=predictions)
    print(f"Saved ({len(predictions)} predictions)")


if __name__ == "__main__":
    main()
