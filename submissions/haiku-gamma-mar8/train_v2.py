#!/usr/bin/env python3
"""Gamma/hadron classifier - simplified version."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GammaDataset(Dataset):
    """Load gamma/hadron training data."""

    def __init__(self, split: str):
        self.matrices = np.load(
            f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/matrices.npy",
            mmap_mode="r",
        )
        self.features = np.load(
            f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/features.npy",
            mmap_mode="r",
        )
        self.labels = np.load(
            f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/labels_gamma.npy",
            mmap_mode="r",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32).flatten()  # 512
        feat = self.features[idx].astype(np.float32)  # 5
        x = np.concatenate([mat, feat])  # 517
        y = int(self.labels[idx])  # 0 or 1
        return torch.from_numpy(x), y


class GammaDNN(nn.Module):
    """Simple DNN classifier."""

    def __init__(self, input_dim=517, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 256)
        self.fc4 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_ds = GammaDataset("train")
    test_ds = GammaDataset("test")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")

    # Data loaders
    train_loader = DataLoader(
        train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = GammaDNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Compute class weights
    train_labels = np.array(train_ds.labels[:])
    n_gamma = (train_labels == 0).sum()
    n_hadron = (train_labels == 1).sum()
    w_gamma = len(train_labels) / (2 * n_gamma)
    w_hadron = len(train_labels) / (2 * n_hadron)
    weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training
    best_survival = float("inf")
    best_scores = None

    print("\nTraining...")
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)

        scheduler.step()

        # Evaluate
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                probs = torch.softmax(logits, dim=1)
                gamma_scores = probs[:, 0]

                test_loss += loss.item() * len(y)
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += len(y)

                all_scores.append(gamma_scores.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        # Compute survival at 99% gamma efficiency
        is_gamma = labels == 0
        is_hadron = labels == 1
        ng = is_gamma.sum()
        nh = is_hadron.sum()

        if ng > 0 and nh > 0:
            sg = np.sort(scores[is_gamma])[::-1]
            idx = int(np.ceil(0.99 * ng)) - 1
            thr = sg[min(idx, ng - 1)]
            n_surv = (scores[is_hadron] >= thr).sum()
            survival = n_surv / nh
        else:
            survival = float("inf")

        if survival < best_survival:
            best_survival = survival
            best_scores = scores

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"E{epoch+1:2d} | loss={train_loss/train_total:.4f} "
            f"acc={train_correct/train_total:.4f} | "
            f"survival@99={survival:.2e} | lr={lr:.2e}"
        )

    print(f"\nBest survival: {best_survival:.2e}")

    # Save results
    output_dir = "/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar8"
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        f"{output_dir}/predictions.npz",
        gamma_scores=best_scores,
    )
    print(f"Saved to {output_dir}/predictions.npz")


if __name__ == "__main__":
    main()
