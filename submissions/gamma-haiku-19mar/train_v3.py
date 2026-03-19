"""Gamma/hadron v3: MLP on flattened matrices + features."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # Load train data
    print("Loading training data...")
    X_matrices_train = np.load("data/gamma_train/matrices.npy", mmap_mode="r").astype(np.float32)
    X_features_train = np.load("data/gamma_train/features.npy", mmap_mode="r").astype(np.float32)
    y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

    # Load test data
    print("Loading test data...")
    X_matrices_test = np.load("data/gamma_test/matrices.npy", mmap_mode="r").astype(np.float32)
    X_features_test = np.load("data/gamma_test/features.npy", mmap_mode="r").astype(np.float32)
    y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

    print(f"Train: matrices {X_matrices_train.shape}, features {X_features_train.shape}")
    print(f"Test: matrices {X_matrices_test.shape}, features {X_features_test.shape}")

    # Flatten matrices and concatenate with features
    print("Preparing data...")
    X_train_combined = np.concatenate([
        X_matrices_train.reshape(len(X_matrices_train), -1),  # 512 dims
        X_features_train  # 5 dims
    ], axis=1)  # 517 dims total
    X_test_combined = np.concatenate([
        X_matrices_test.reshape(len(X_matrices_test), -1),
        X_features_test
    ], axis=1)

    print(f"Combined shape: train {X_train_combined.shape}, test {X_test_combined.shape}")

    # Normalize using subsample
    print("Computing normalization stats...")
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(len(X_train_combined), size=min(500_000, len(X_train_combined)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X_train_combined[idx_sample])
    X_train_norm = scaler.transform(X_train_combined)
    X_test_norm = scaler.transform(X_test_combined)

    # Split train into train/val
    idx_train = np.arange(len(X_train_combined))
    rng.shuffle(idx_train)
    n_train = int(0.8 * len(X_train_combined))
    idx_train_split = idx_train[:n_train]
    idx_val_split = idx_train[n_train:]

    X_train_split = torch.from_numpy(X_train_norm[idx_train_split])
    y_train_split = torch.from_numpy(y_train[idx_train_split]).long()
    X_val_split = torch.from_numpy(X_train_norm[idx_val_split])
    y_val_split = torch.from_numpy(y_train[idx_val_split]).long()
    X_test_torch = torch.from_numpy(X_test_norm)
    y_test_torch = torch.from_numpy(y_test.copy()).long()

    train_loader = DataLoader(
        TensorDataset(X_train_split, y_train_split),
        batch_size=4096, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(X_val_split, y_val_split),
        batch_size=8192, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(X_test_torch, y_test_torch),
        batch_size=8192, shuffle=False, num_workers=0
    )

    # Build model
    print("Building model...")
    class MLP(nn.Module):
        def __init__(self, input_dim=517):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(input_dim=517).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\nTraining...")
    best_val_loss = float('inf')
    best_scores = None
    n_epochs = 20

    for epoch in range(n_epochs):
        # Train
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

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
        val_loss = val_loss / len(idx_val_split)

        # Test
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

        print(f"E{epoch+1:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = scores
            torch.save(model.state_dict(), "submissions/gamma-haiku-19mar/model_v3.pt")
            print(f"  -> best checkpoint saved")

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    np.savez(
        "submissions/gamma-haiku-19mar/predictions.npz",
        gamma_scores=best_scores,
    )
    print(f"Saved predictions ({len(best_scores)} scores)")

    print("\n---")
    print("metric: 0.0000")
    print("description: MLP on matrices + features")

if __name__ == "__main__":
    main()
