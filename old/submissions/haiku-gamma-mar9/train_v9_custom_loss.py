"""Custom loss that directly optimizes hadronic survival @ 99% gamma efficiency.

This is tricky because the metric itself is not differentiable. But we can:
1. Use soft approximations
2. Or use the DNN to generate scores, then post-process to optimize the metric
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
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def focal_loss(logits, labels, alpha=2.0, gamma=2.0):
    """Focal loss to handle class imbalance better."""
    criterion = nn.CrossEntropyLoss(reduction='none')
    ce_loss = criterion(logits, labels)
    probs = torch.softmax(logits, dim=1)
    pt = torch.gather(probs, 1, labels.unsqueeze(-1)).squeeze(-1)
    focal_loss_val = (1 - pt) ** gamma * ce_loss
    return focal_loss_val.mean()


def compute_survival_at_99(gamma_probs, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(gamma_probs[is_gamma])
    ng = len(sg)
    thr_99 = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
    n_hadron_surviving = (gamma_probs[is_hadron] >= thr_99).sum()
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

    # Class weights
    train_labels_all = raw_train.labels[train_ds.indices]
    n_gamma_train = (train_labels_all == 0).sum()
    n_hadron_train = (train_labels_all == 1).sum()
    w_gamma = n_hadron_train / (2 * n_gamma_train)
    w_hadron = n_gamma_train / (2 * n_hadron_train)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    print("Training with focal loss...")
    best_survival = 1.0
    patience = 20

    for epoch in range(60):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = focal_loss(logits, y, gamma=2.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Eval
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs[:, 0].cpu().numpy())
                all_labels.append(y.numpy())

        val_probs = np.concatenate(all_probs)
        val_labels = np.concatenate(all_labels)
        val_survival, _ = compute_survival_at_99(val_probs, val_labels)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"val_surv@99={val_survival:.4f} lr={lr:.5f}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 20
            torch.save(model.state_dict(),
                      "submissions/haiku-gamma-mar9/model_best_v9.pt")
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stop at epoch {epoch+1}")
                break

    # Test
    print("\nEvaluating on test set...")
    model.load_state_dict(
        torch.load("submissions/haiku-gamma-mar9/model_best_v9.pt")
    )
    model.eval()

    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, 0].cpu().numpy())

    test_probs = np.concatenate(all_probs)
    test_survival, test_thr = compute_survival_at_99(test_probs, test_labels)

    print(f"Threshold: {test_thr:.4f}")
    print(f"Test survival @ 99% gamma eff: {test_survival:.4f}")

    np.savez("submissions/haiku-gamma-mar9/predictions_v9.npz",
             gamma_scores=test_probs)


if __name__ == "__main__":
    main()
