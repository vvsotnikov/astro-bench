"""v3: Muon-focused with engineered features

Key physics insights:
- Muon content is THE key differentiator
- Extract spatial statistics from muon channel: mean, max, sum, std
- Combine with scalar features for highly focused classification
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from load_data import load_train, load_test
from verify import evaluate


class EngineeredDataset(Dataset):
    def __init__(self, matrices, features, labels):
        # Extract muon channel and compute spatial statistics
        muon_channel = matrices[:, :, :, 1]  # (N, 16, 16)

        muon_stats = np.stack([
            muon_channel.reshape(len(matrices), -1).mean(axis=1),  # mean
            muon_channel.reshape(len(matrices), -1).max(axis=1),   # max
            muon_channel.reshape(len(matrices), -1).sum(axis=1),   # sum
            muon_channel.reshape(len(matrices), -1).std(axis=1),   # std
        ], axis=1).astype(np.float32)

        # Electron channel stats
        electron_channel = matrices[:, :, :, 0]
        electron_stats = np.stack([
            electron_channel.reshape(len(matrices), -1).mean(axis=1),
            electron_channel.reshape(len(matrices), -1).max(axis=1),
        ], axis=1).astype(np.float32)

        # Combine everything: muon_stats (4) + electron_stats (2) + features (5) = 11 total
        self.combined_features = np.concatenate([
            muon_stats, electron_stats, features.astype(np.float32)
        ], axis=1)

        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.combined_features[idx]), self.labels[idx]


class EngineeredMLP(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
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

    # Create datasets with engineered features
    print("Engineering features...")
    train_ds = EngineeredDataset(X_train, f_train, y_train)
    test_ds = EngineeredDataset(X_test, f_test, y_test)

    train_size = int(0.95 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds_split, val_ds_split = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Normalize features using train split stats
    train_features = train_ds_split.dataset.combined_features[train_ds_split.indices]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0

    # Apply normalization
    train_ds_split.dataset.combined_features = (train_ds_split.dataset.combined_features - mean) / std
    test_ds.combined_features = (test_ds.combined_features - mean) / std

    train_loader = DataLoader(train_ds_split, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds_split, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    model = EngineeredMLP(input_dim=11).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    n_gamma = int((y_train == 0).sum())
    n_hadron = int((y_train == 1).sum())
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    for epoch in range(40):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        scheduler.step()

        model.eval()
        val_loss = 0
        all_scores = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

        epoch_val_loss = val_loss / val_size
        scores = np.concatenate(all_scores)

        print(f"Epoch {epoch+1:2d}/40: train_loss={total_loss/train_size:.4f} "
              f"val_loss={epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_scores = scores
            torch.save(model.state_dict(), "model_v3.pt")

    # Test set evaluation
    model.load_state_dict(torch.load("model_v3.pt"))
    model.eval()
    test_scores = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)
    evaluate(test_scores, "v3: Engineered muon features (mean/max/sum/std), 40 epochs")


if __name__ == "__main__":
    main()
