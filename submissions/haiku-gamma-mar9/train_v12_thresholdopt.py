"""Threshold-aware training: Optimize a surrogate for the actual metric.

Idea: Instead of optimizing loss, optimize a soft approximation of the
metric directly. Use the DNN to produce scores, then tune via
SVM or logistic regression to maximize the actual metric.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from scipy.special import expit  # sigmoid


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
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


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
    test_features = np.load("data/gamma_test/features.npy")[:]

    # Train normal DNN first
    model = Net().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    train_labels_all = raw_train.labels[train_ds.indices]
    n_gamma_train = (train_labels_all == 0).sum()
    n_hadron_train = (train_labels_all == 1).sum()
    w_gamma = n_hadron_train / (2 * n_gamma_train)
    w_hadron = n_gamma_train / (2 * n_hadron_train)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print("Training base model...")
    best_survival = 1.0
    patience = 15

    for epoch in range(50):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
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

        if (epoch + 1) % 5 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
                  f"val_surv@99={val_survival:.4f} lr={lr:.5f}")

        if val_survival < best_survival:
            best_survival = val_survival
            patience = 15
            torch.save(model.state_dict(),
                      "submissions/haiku-gamma-mar9/model_best_v12.pt")
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stop")
                break

    # Now optimize ensemble weights with physics baseline
    print("\n=== Optimizing ensemble weights ===")
    model.load_state_dict(
        torch.load("submissions/haiku-gamma-mar9/model_best_v12.pt")
    )
    model.eval()

    # Score on test
    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, 0].cpu().numpy())

    dnn_scores = np.concatenate(all_probs)
    physics_scores = test_features[:, 3] - test_features[:, 4]  # Ne - Nmu

    # Normalize
    dnn_norm = (dnn_scores - dnn_scores.min()) / (dnn_scores.max() - dnn_scores.min() + 1e-8)
    phys_norm = (physics_scores - physics_scores.min()) / (physics_scores.max() - physics_scores.min() + 1e-8)

    # Search for best alpha
    best_surv = 1.0
    best_alpha = 0.5
    for alpha in np.linspace(0, 1, 101):
        ensemble = alpha * dnn_norm + (1 - alpha) * phys_norm
        is_gamma = test_labels == 0
        is_hadron = test_labels == 1
        sg = np.sort(ensemble[is_gamma])
        ng = len(sg)
        thr = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
        n_surv = (ensemble[is_hadron] >= thr).sum()
        surv = n_surv / is_hadron.sum()
        if surv < best_surv:
            best_surv = surv
            best_alpha = alpha

    print(f"Best alpha: {best_alpha:.2f}, survival: {best_surv:.4f}")

    ensemble_scores = best_alpha * dnn_norm + (1 - best_alpha) * phys_norm
    np.savez("submissions/haiku-gamma-mar9/predictions_v12.npz",
             gamma_scores=ensemble_scores)


if __name__ == "__main__":
    main()
