"""Margin loss for gamma/hadron separation.

Use a loss that explicitly pushes gamma scores up and hadron scores down,
similar to what the published RF approach might be doing.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim


class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].flatten().astype(np.float32)
        feat = self.features[idx].astype(np.float32)
        x = np.concatenate([mat, feat])
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(x), int(self.labels[idx])


def compute_stats(dataset, n_samples=500_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    samples = []
    for idx in indices:
        mat = dataset.matrices[idx].flatten().astype(np.float32)
        feat = dataset.features[idx].astype(np.float32)
        samples.append(np.concatenate([mat, feat]))
    samples = np.stack(samples)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class Net(nn.Module):
    def __init__(self, input_dim=517, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def margin_loss(scores, labels, margin=0.5):
    """Margin loss: maximize gap between gamma and hadron scores."""
    is_gamma = labels == 0
    is_hadron = labels == 1

    # Want: gamma scores >> hadron scores
    # Loss increases when gamma < hadron + margin
    gamma_scores = scores[is_gamma]
    hadron_scores = scores[is_hadron]

    # Compute pairwise margins
    # For each hadron, find the closest gamma
    loss = 0
    for h_score in hadron_scores:
        # Margin: want gamma > hadron + margin
        margin_loss_h = torch.max(
            torch.tensor(0.0, device=scores.device),
            h_score - torch.min(gamma_scores) + margin
        )
        loss = loss + margin_loss_h

    return loss / len(hadron_scores)


def compute_survival_at_99(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr_99 = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
    n_hadron_surviving = (scores[is_hadron] >= thr_99).sum()
    survival_99 = n_hadron_surviving / is_hadron.sum()
    return survival_99, thr_99


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}\n")

    print("Loading data...")
    raw_train = GammaDataset("train")
    print(f"  Total: {len(raw_train)} events")

    print("Computing normalization...")
    mean, std = compute_stats(raw_train)

    # Split
    n_train = int(0.8 * len(raw_train))
    n_val = len(raw_train) - n_train
    train_ds, val_ds = random_split(
        GammaDataset("train", mean=mean, std=std),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(2026)
    )

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True,
                             num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False,
                           num_workers=8, pin_memory=True)

    test_ds = GammaDataset("test", mean=mean, std=std)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

    model = Net().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print("Training with margin loss...")
    best_survival = 1.0
    patience = 20

    for epoch in range(50):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = margin_loss(scores, y, margin=0.3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Eval
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                scores = model(x)
                all_scores.append(scores.cpu().numpy())
                all_labels.append(y.numpy())

        val_scores = np.concatenate(all_scores)
        val_labels = np.concatenate(all_labels)
        val_survival, _ = compute_survival_at_99(val_scores, val_labels)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"val_surv@99={val_survival:.4f} lr={lr:.5f}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 20
            torch.save(model.state_dict(),
                      "submissions/haiku-gamma-mar9/model_best_v8.pt")
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stop at epoch {epoch+1}")
                break

    # Test
    print("\nEvaluating on test set...")
    model.load_state_dict(
        torch.load("submissions/haiku-gamma-mar9/model_best_v8.pt")
    )
    model.eval()

    all_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            scores = model(x)
            all_scores.append(scores.cpu().numpy())

    test_scores = np.concatenate(all_scores)
    test_survival, test_thr = compute_survival_at_99(test_scores, test_labels)

    print(f"Threshold: {test_thr:.4f}")
    print(f"Test survival @ 99% gamma eff: {test_survival:.4f}")

    np.savez("submissions/haiku-gamma-mar9/predictions_v8.npz",
             gamma_scores=test_scores)


if __name__ == "__main__":
    main()
