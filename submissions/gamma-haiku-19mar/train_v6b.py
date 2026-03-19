"""Gamma/hadron v6b: Deeper MLP with explicit output flushing."""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}", flush=True)

    # Load data
    print("Loading data...", flush=True)
    X_matrices_train = np.load("data/gamma_train/matrices.npy", mmap_mode="r").astype(np.float32)
    X_features_train = np.load("data/gamma_train/features.npy", mmap_mode="r").astype(np.float32)
    y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

    X_matrices_test = np.load("data/gamma_test/matrices.npy", mmap_mode="r").astype(np.float32)
    X_features_test = np.load("data/gamma_test/features.npy", mmap_mode="r").astype(np.float32)
    y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

    print(f"Train: matrices {X_matrices_train.shape}, features {X_features_train.shape}", flush=True)
    print(f"Test: matrices {X_matrices_test.shape}, features {X_features_test.shape}", flush=True)

    # Flatten and combine
    print("Preparing data...", flush=True)
    X_train_combined = np.concatenate([
        X_matrices_train.reshape(len(X_matrices_train), -1),
        X_features_train
    ], axis=1)
    X_test_combined = np.concatenate([
        X_matrices_test.reshape(len(X_matrices_test), -1),
        X_features_test
    ], axis=1)

    # Normalize
    print("Normalizing...", flush=True)
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(len(X_train_combined), size=min(500_000, len(X_train_combined)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X_train_combined[idx_sample])
    X_train_norm = scaler.transform(X_train_combined)
    X_test_norm = scaler.transform(X_test_combined)

    # Train/val split
    idx_train = np.arange(len(X_train_combined))
    rng.shuffle(idx_train)
    n_train = int(0.8 * len(X_train_combined))
    idx_train_split = idx_train[:n_train]
    idx_val_split = idx_train[n_train:]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_norm[idx_train_split]),
            torch.from_numpy(y_train[idx_train_split]).long()
        ),
        batch_size=4096, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_norm[idx_val_split]),
            torch.from_numpy(y_train[idx_val_split]).long()
        ),
        batch_size=8192, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test_norm),
            torch.from_numpy(y_test.copy()).long()
        ),
        batch_size=8192, shuffle=False, num_workers=0
    )

    # Deeper MLP
    print("Building model...", flush=True)
    class DeepMLP(nn.Module):
        def __init__(self, input_dim=517):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ELU(),
                nn.Dropout(0.25),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ELU(),
                nn.Dropout(0.25),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            return self.net(x)

    model = DeepMLP().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Class weights
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Training...", flush=True)
    best_val_loss = float('inf')
    best_scores = None
    n_epochs = 25

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_loss = total_loss / len(idx_train_split)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
        val_loss = val_loss / len(idx_val_split)

        all_scores = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_scores.append(probs[:, 0].cpu().numpy())
        scores = np.concatenate(all_scores)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"E{epoch+1:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = scores
            torch.save(model.state_dict(), "submissions/gamma-haiku-19mar/model_v6b.pt")
            print(f"  -> best checkpoint saved", flush=True)

    print(f"\nBest val_loss: {best_val_loss:.4f}", flush=True)
    np.savez(
        "submissions/gamma-haiku-19mar/predictions.npz",
        gamma_scores=best_scores,
    )
    print(f"Saved predictions ({len(best_scores)} scores)", flush=True)

    print("\n---", flush=True)
    print("metric: 0.0000", flush=True)
    print("description: Deeper MLP (4 hidden layers)", flush=True)

if __name__ == "__main__":
    main()
