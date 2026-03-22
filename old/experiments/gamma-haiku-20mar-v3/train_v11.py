"""v11: MuonCNN with BCELoss (binary cross entropy) instead of multiclass loss

BCELoss treats it as a pure binary problem, which might work better for this case.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from load_data import load_train, load_test
from verify import evaluate


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels):
        self.matrices = torch.from_numpy(matrices.astype(np.float32)).permute(0, 3, 1, 2)
        self.features = torch.from_numpy(features.astype(np.float32))
        # Convert labels to float for BCE: 0 (gamma) -> 0.0, 1 (hadron) -> 1.0
        self.labels = torch.from_numpy((labels == 1).astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.matrices[idx], self.features[idx], self.labels[idx]


class MuonCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.muon_path = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )

        self.electron_path = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )

        muon_feat = 64 * 4 * 4
        electron_feat = 16 * 4 * 4

        self.fusion = nn.Sequential(
            nn.Linear(muon_feat + electron_feat + 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, matrices, features):
        muon = matrices[:, 1:2, :, :]
        electron = matrices[:, 0:1, :, :]

        muon_feat = self.muon_path(muon).flatten(1)
        electron_feat = self.electron_path(electron).flatten(1)

        combined = torch.cat([muon_feat, electron_feat, features], dim=1)
        return self.fusion(combined).squeeze(1)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()

    train_ds = GammaDataset(X_train, f_train, y_train)
    test_ds = GammaDataset(X_test, f_test, y_test)

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

    # Class weights for BCE
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(device)
    print(f"Positive weight: {pos_weight.item():.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_scores = None
    best_val_loss = float("inf")

    for epoch in range(30):
        model.train()
        total_loss = 0
        for matrices, features, labels in train_loader:
            matrices, features, labels = matrices.to(device), features.to(device), labels.to(device)
            logits = model(matrices, features)
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
            for matrices, features, labels in val_loader:
                matrices, features, labels = matrices.to(device), features.to(device), labels.to(device)
                logits = model(matrices, features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                # For evaluation, convert sigmoid(logits) to gamma probability (1 - sigmoid = gamma prob)
                probs = torch.sigmoid(logits)
                all_scores.append((1 - probs).cpu().numpy())

        epoch_val_loss = val_loss / val_size
        scores = np.concatenate(all_scores)

        print(f"Epoch {epoch+1:2d}/30: train_loss={total_loss/train_size:.4f} "
              f"val_loss={epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_scores = scores
            torch.save(model.state_dict(), "model_v11.pt")

    model.load_state_dict(torch.load("model_v11.pt"))
    model.eval()
    test_scores = []
    with torch.no_grad():
        for matrices, features, labels in test_loader:
            matrices, features = matrices.to(device), features.to(device)
            logits = model(matrices, features)
            probs = torch.sigmoid(logits)
            test_scores.append((1 - probs).cpu().numpy())

    test_scores = np.concatenate(test_scores)
    evaluate(test_scores, "v11: MuonCNN with BCEWithLogitsLoss, 30 epochs")


if __name__ == "__main__":
    main()
