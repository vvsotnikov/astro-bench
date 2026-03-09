"""Train with mixed loss: classification + regression."""

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
        label = int(self.labels[idx])
        return torch.from_numpy(x), label


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


class MixedDNN(nn.Module):
    """DNN with both classification and regression outputs."""
    def __init__(self, input_dim=517, hidden=512):
        super().__init__()
        self.shared = nn.Sequential(
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
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden // 2, 2),  # Classification
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden // 2, 1),  # Regression
            nn.Sigmoid()
        )

    def forward(self, x):
        shared_feat = self.shared(x)
        logits = self.class_head(shared_feat)
        score = self.reg_head(shared_feat).squeeze(-1)
        return logits, score


def compute_survival_at_75(gamma_scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(gamma_scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (gamma_scores[is_hadron] >= thr).sum()
    return n_hadron_surviving / is_hadron.sum()


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}\n")

    # Load and prepare data
    print("Loading data...")
    raw_train = GammaDataset("train")
    print(f"  Training: {len(raw_train)} events")

    print("Computing normalization...")
    mean, std = compute_stats(raw_train)

    # Train/val split
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

    # Train DNN
    model = MixedDNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    train_labels_all = raw_train.labels[train_ds.indices]
    n_gamma_train = (train_labels_all == 0).sum()
    n_hadron_train = (train_labels_all == 1).sum()
    w_gamma = n_hadron_train / (2 * n_gamma_train)
    w_hadron = n_gamma_train / (2 * n_hadron_train)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    bce_loss = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print("Training mixed loss DNN...")
    best_survival = 1.0
    patience = 15

    for epoch in range(50):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)

            logits, scores = model(x)

            # Classification loss
            loss_ce = ce_loss(logits, y.long())

            # Regression loss (score should be 1 for gamma, 0 for hadron)
            target_score = (y == 0).float()
            loss_bce = bce_loss(scores, target_score)

            # Combined loss (equal weight)
            loss = 0.5 * loss_ce + 0.5 * loss_bce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Eval
        if epoch % 2 == 0:
            model.eval()
            all_scores = []
            all_labels = []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    _, scores = model(x)
                    all_scores.append(scores.cpu().numpy())
                    all_labels.append(y.numpy())

            val_scores = np.concatenate(all_scores)
            val_labels = np.concatenate(all_labels)
            val_survival = compute_survival_at_75(val_scores, val_labels)

            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
                  f"val_surv@75={val_survival:.4e} lr={lr:.6f}")

            if val_survival < best_survival:
                best_survival = val_survival
                patience = 15
                torch.save(model.state_dict(), "submissions/haiku-gamma-mar9-v2/model_best_v17.pt")
            else:
                patience -= 1
                if patience <= 0:
                    print(f"Early stopping")
                    break

    # Inference on test
    print("\nInference on test set...")
    model.load_state_dict(torch.load("submissions/haiku-gamma-mar9-v2/model_best_v17.pt"))
    model.eval()

    all_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, scores = model(x)
            all_scores.append(scores.cpu().numpy())

    test_scores = np.concatenate(all_scores)

    # Save predictions
    np.savez("submissions/haiku-gamma-mar9-v2/predictions_v17.npz",
             gamma_scores=test_scores)

    # Compute final score
    test_survival = compute_survival_at_75(test_scores, test_labels)
    print(f"Test survival @ 75% gamma eff: {test_survival:.4e}")

    print(f"\n---")
    print(f"metric: {test_survival:.4e}")
    print(f"description: Mixed loss (0.5 CrossEntropy + 0.5 BCELoss)")


if __name__ == "__main__":
    main()
