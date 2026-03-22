"""v18: Very deep pure MLP on flattened matrices + features

No convolutions, just fully connected layers. Test if spatial structure matters.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from load_data import load_train, load_test
from verify import evaluate


class FlatDataset(Dataset):
    def __init__(self, matrices, features, labels):
        # Flatten matrices to (N, 512)
        self.flat_matrices = torch.from_numpy(matrices.astype(np.float32).reshape(len(matrices), -1))
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Concatenate: (512 + 5 = 517 total features)
        combined = torch.cat([self.flat_matrices[idx], self.features[idx]])
        return combined, self.labels[idx]


class DeepMLP(nn.Module):
    def __init__(self, input_dim=517):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()

    train_ds = FlatDataset(X_train, f_train, y_train)
    test_ds = FlatDataset(X_test, f_test, y_test)

    train_size = int(0.95 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds_split, val_ds_split = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds_split, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds_split, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = DeepMLP(input_dim=517).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    n_gamma = int((y_train == 0).sum())
    n_hadron = int((y_train == 1).sum())
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_scores = None
    best_val_loss = float("inf")

    for epoch in range(30):
        model.train()
        total_loss = 0
        for combined, labels in train_loader:
            combined, labels = combined.to(device), labels.to(device)
            logits = model(combined)
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
            for combined, labels in val_loader:
                combined, labels = combined.to(device), labels.to(device)
                logits = model(combined)
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
            torch.save(model.state_dict(), "model_v18.pt")

    model.load_state_dict(torch.load("model_v18.pt"))
    model.eval()
    test_scores = []
    with torch.no_grad():
        for combined, labels in test_loader:
            combined = combined.to(device)
            logits = model(combined)
            test_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())

    test_scores = np.concatenate(test_scores)
    evaluate(test_scores, "v18: DeepMLP (5 layers, flattened input), 30 epochs")


if __name__ == "__main__":
    main()
