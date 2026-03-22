"""v7: Lightweight model - Linear on top of spatial pooled features

Simpler baseline to see if we're overfitting with deeper models.
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
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.matrices[idx], self.features[idx], self.labels[idx]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Very lightweight: just pool and classify
        self.muon_pool = nn.AdaptiveAvgPool2d(2)
        self.electron_pool = nn.AdaptiveAvgPool2d(2)

        # Global statistics
        muon_feat = 1 * 2 * 2
        electron_feat = 1 * 2 * 2

        self.classifier = nn.Sequential(
            nn.Linear(muon_feat + electron_feat + 5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, matrices, features):
        muon = matrices[:, 1:2, :, :]
        electron = matrices[:, 0:1, :, :]

        muon_feat = self.muon_pool(muon).flatten(1)
        electron_feat = self.electron_pool(electron).flatten(1)

        combined = torch.cat([muon_feat, electron_feat, features], dim=1)
        return self.classifier(combined)


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

    train_loader = DataLoader(train_ds_split, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds_split, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    model = SimpleCNN().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    n_gamma = int((y_train == 0).sum())
    n_hadron = int((y_train == 1).sum())
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    for epoch in range(40):
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
                all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

        epoch_val_loss = val_loss / val_size
        scores = np.concatenate(all_scores)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/40: train_loss={total_loss/train_size:.4f} val_loss={epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_scores = scores
            torch.save(model.state_dict(), "model_v7.pt")

    model.load_state_dict(torch.load("model_v7.pt"))
    model.eval()
    test_scores = []
    with torch.no_grad():
        for matrices, features, labels in test_loader:
            matrices, features = matrices.to(device), features.to(device)
            logits = model(matrices, features)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)
    evaluate(test_scores, "v7: SimpleCNN (lightweight pooling baseline), 40 epochs")


if __name__ == "__main__":
    main()
