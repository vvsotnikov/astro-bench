"""Gamma/hadron v4: CNN on matrices + MLP on features.

Uses spatial structure of detector matrices via convolutions.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class CNNMLP(nn.Module):
    """CNN on matrices + MLP on features, fused before final layer."""
    def __init__(self):
        super().__init__()
        # CNN branch: 16x16x2 matrices
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # 2x2
        )
        cnn_out_dim = 64 * 2 * 2  # 256

        # Feature MLP branch: 5 features
        self.feat_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Fusion
        fusion_dim = cnn_out_dim + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x_matrix, x_feat):
        # CNN branch
        cnn_out = self.conv(x_matrix)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        # Feature branch
        feat_out = self.feat_mlp(x_feat)

        # Fuse
        fused = torch.cat([cnn_out, feat_out], dim=1)
        logits = self.fusion(fused)
        return logits


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # Load data
    print("Loading training data...")
    X_matrices_train = np.load("data/gamma_train/matrices.npy", mmap_mode="r").astype(np.float32)
    X_features_train = np.load("data/gamma_train/features.npy", mmap_mode="r").astype(np.float32)
    y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

    print("Loading test data...")
    X_matrices_test = np.load("data/gamma_test/matrices.npy", mmap_mode="r").astype(np.float32)
    X_features_test = np.load("data/gamma_test/features.npy", mmap_mode="r").astype(np.float32)
    y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

    print(f"Train: {X_matrices_train.shape}, {X_features_train.shape}, {y_train.shape}")
    print(f"Test: {X_matrices_test.shape}, {X_features_test.shape}, {y_test.shape}")

    # Normalize features
    print("Normalizing features...")
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(len(X_features_train), size=min(500_000, len(X_features_train)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X_features_train[idx_sample])
    X_features_train = scaler.transform(X_features_train)
    X_features_test = scaler.transform(X_features_test)

    # Normalize matrices (subtract mean, divide by std - per-channel)
    X_matrices_train_flat = X_matrices_train.reshape(-1, 2)
    m = X_matrices_train_flat.mean(axis=0)
    s = X_matrices_train_flat.std(axis=0)
    s[s == 0] = 1.0
    X_matrices_train = (X_matrices_train - m) / s
    X_matrices_test = (X_matrices_test - m) / s

    # Convert to torch (CHW format for conv)
    X_matrices_train = torch.from_numpy(X_matrices_train.transpose(0, 3, 1, 2))  # (N, 2, 16, 16)
    X_features_train = torch.from_numpy(X_features_train)
    X_matrices_test = torch.from_numpy(X_matrices_test.transpose(0, 3, 1, 2))
    X_features_test = torch.from_numpy(X_features_test)

    # Split into train/val
    idx_train = np.arange(len(y_train))
    rng.shuffle(idx_train)
    n_train = int(0.8 * len(y_train))
    idx_train_split = idx_train[:n_train]
    idx_val_split = idx_train[n_train:]

    train_loader = DataLoader(
        TensorDataset(
            X_matrices_train[idx_train_split],
            X_features_train[idx_train_split],
            torch.from_numpy(y_train[idx_train_split]).long()
        ),
        batch_size=2048, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        TensorDataset(
            X_matrices_train[idx_val_split],
            X_features_train[idx_val_split],
            torch.from_numpy(y_train[idx_val_split]).long()
        ),
        batch_size=4096, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        TensorDataset(X_matrices_test, X_features_test, torch.from_numpy(y_test.copy()).long()),
        batch_size=4096, shuffle=False, num_workers=0
    )

    # Model
    print("Building CNN+MLP model...")
    model = CNNMLP().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\nTraining...")
    best_val_loss = float('inf')
    best_scores = None
    n_epochs = 15

    for epoch in range(n_epochs):
        # Train
        model.train()
        total_loss = 0
        for x_mat, x_feat, y in train_loader:
            x_mat, x_feat, y = x_mat.to(device), x_feat.to(device), y.to(device)
            logits = model(x_mat, x_feat)
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
            for x_mat, x_feat, y in val_loader:
                x_mat, x_feat, y = x_mat.to(device), x_feat.to(device), y.to(device)
                logits = model(x_mat, x_feat)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
        val_loss = val_loss / len(idx_val_split)

        # Test
        all_scores = []
        with torch.no_grad():
            for x_mat, x_feat, y in test_loader:
                x_mat, x_feat = x_mat.to(device), x_feat.to(device)
                logits = model(x_mat, x_feat)
                probs = torch.softmax(logits, dim=1)
                all_scores.append(probs[:, 0].cpu().numpy())
        scores = np.concatenate(all_scores)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"E{epoch+1:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scores = scores
            torch.save(model.state_dict(), "submissions/gamma-haiku-19mar/model_v4.pt")
            print(f"  -> best checkpoint saved")

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    np.savez(
        "submissions/gamma-haiku-19mar/predictions.npz",
        gamma_scores=best_scores,
    )
    print(f"Saved predictions ({len(best_scores)} scores)")

    print("\n---")
    print("metric: 0.0000")
    print("description: CNN+MLP with feature fusion")

if __name__ == "__main__":
    main()
