"""v1: CNN on muon channel + features, physics-informed

Key insight: Gamma rays have almost NO muons (~3.0), hadrons have more (~3.5).
We should heavily weight the muon channel (channel 1) and use features as auxiliary.

Architecture:
- Separate paths for muon channel vs electron channel
- Muon path: 2 conv layers with larger capacity
- Electron path: 1 conv layer
- Merge and classify
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from load_data import load_train, load_test
from verify import evaluate


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels):
        # Reshape matrices from (N, 16, 16, 2) to (N, 2, 16, 16) for PyTorch Conv2d
        self.matrices = torch.from_numpy(matrices.astype(np.float32)).permute(0, 3, 1, 2)
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.matrices[idx], self.features[idx], self.labels[idx]


class MuonCNN(nn.Module):
    """Physics-informed CNN: separate muon and electron channels"""
    def __init__(self):
        super().__init__()
        # Muon path (channel 1) — main signal
        self.muon_path = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )

        # Electron path (channel 0) — secondary info
        self.electron_path = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )

        # Fusion
        muon_feat = 64 * 4 * 4
        electron_feat = 16 * 4 * 4
        feat_size = 5

        self.fusion = nn.Sequential(
            nn.Linear(muon_feat + electron_feat + feat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, matrices, features):
        # matrices: (B, 2, 16, 16) after reshape in dataset
        # Split channels
        muon = matrices[:, 1:2, :, :]  # (B, 1, 16, 16)
        electron = matrices[:, 0:1, :, :]  # (B, 1, 16, 16)

        # Process paths
        muon_feat = self.muon_path(muon).flatten(1)
        electron_feat = self.electron_path(electron).flatten(1)

        # Concatenate all features
        combined = torch.cat([muon_feat, electron_feat, features], dim=1)
        logits = self.fusion(combined)
        return logits


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")
    print(f"  Train gamma: {(y_train==0).sum()}, hadron: {(y_train==1).sum()}")

    # Create datasets
    train_ds = GammaDataset(X_train, f_train, y_train)
    test_ds = GammaDataset(X_test, f_test, y_test)

    # Use validation set from train data
    train_size = int(0.95 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds_split, val_ds_split = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds_split, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds_split, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = MuonCNN().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for imbalanced data
    n_gamma = int((y_train == 0).sum())
    n_hadron = int((y_train == 1).sum())
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    for epoch in range(30):
        # Train
        model.train()
        total_loss = 0
        for matrices, features, labels in train_loader:
            matrices = matrices.to(device)
            features = features.to(device)
            labels = labels.to(device)

            logits = model(matrices, features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        all_scores = []
        with torch.no_grad():
            for matrices, features, labels in val_loader:
                matrices = matrices.to(device)
                features = features.to(device)
                labels = labels.to(device)

                logits = model(matrices, features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)

                all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

        epoch_val_loss = val_loss / val_size
        scores = np.concatenate(all_scores)

        print(f"Epoch {epoch+1:2d}/30: train_loss={total_loss/train_size:.4f} "
              f"val_loss={epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_scores = scores
            torch.save(model.state_dict(), "model_v1.pt")

    # Evaluate on test set using best model
    model.load_state_dict(torch.load("model_v1.pt"))
    model.eval()
    test_scores = []
    with torch.no_grad():
        for matrices, features, labels in test_loader:
            matrices = matrices.to(device)
            features = features.to(device)
            logits = model(matrices, features)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)

    # Official evaluation
    evaluate(test_scores, "v1: MuonCNN (separate paths for muon/electron, 30 epochs)")


if __name__ == "__main__":
    main()
