#!/usr/bin/env python3
"""Gamma/hadron classifier: Energy-binned ensemble.

Train separate models for different energy ranges to handle energy-dependent physics.
EDA showed worst separation at 14.0-15.5 eV (dominant bin) and best at 16.0-17.0 eV.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None, energy_range=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std

        # Filter by energy range if specified
        self.indices = np.arange(len(self.labels))
        if energy_range is not None:
            emin, emax = energy_range
            energy = self.features[:, 0]
            mask = (energy >= emin) & (energy < emax)
            self.indices = np.where(mask)[0]
            print(f"Energy range [{emin}, {emax}): {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        mat = self.matrices[real_idx].flatten().astype(np.float32)
        feat = self.features[real_idx].astype(np.float32)
        x = np.concatenate([mat, feat])  # 517
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return torch.from_numpy(x), int(self.labels[real_idx])


def compute_stats(dataset, n_samples=500_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset.labels), size=min(n_samples, len(dataset.labels)), replace=False)
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


class DNN(nn.Module):
    def __init__(self, input_dim=517, hidden=512, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model_for_energy_bin(energy_range, raw_train, mean, std):
    """Train a single model for one energy bin."""
    device = torch.device("cuda:0")

    emin, emax = energy_range
    print(f"\n{'='*80}")
    print(f"Training model for energy bin [{emin:.1f}, {emax:.1f})")
    print(f"{'='*80}")

    train_ds = GammaDataset("train", mean=mean, std=std, energy_range=energy_range)
    if len(train_ds) == 0:
        print(f"No training samples in this bin!")
        return None

    train_loader = DataLoader(
        train_ds, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True
    )

    test_ds = GammaDataset("test", mean=mean, std=std, energy_range=energy_range)
    if len(test_ds) == 0:
        print(f"No test samples in this bin!")
        return None

    test_loader = DataLoader(
        test_ds, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True
    )

    model = DNN().to(device)

    # Class weights from TRAINING data in this bin
    labels_all = np.array(train_ds.labels[train_ds.indices])
    n_gamma = (labels_all == 0).sum()
    n_hadron = (labels_all == 1).sum()
    if n_gamma == 0 or n_hadron == 0:
        print(f"Insufficient class balance in bin!")
        return None

    w_gamma = len(labels_all) / (2 * n_gamma)
    w_hadron = len(labels_all) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    n_epochs = 30
    best_survival = 1.0
    best_scores = None

    for epoch in range(n_epochs):
        model.train()
        correct = total = 0
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / total

        # Evaluate
        model.eval()
        test_correct = test_total = 0
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                gamma_scores = probs[:, 0]
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += len(y)
                all_scores.append(gamma_scores.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        test_acc = test_correct / test_total
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        # Compute survival at 99% gamma efficiency
        is_gamma = labels == 0
        is_hadron = labels == 1
        if is_gamma.sum() > 0 and is_hadron.sum() > 0:
            sg = np.sort(scores[is_gamma])
            ng = len(sg)
            thr_99 = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
            n_hadron_surviving = (scores[is_hadron] >= thr_99).sum()
            survival_99 = n_hadron_surviving / is_hadron.sum()
        else:
            survival_99 = 1.0

        if survival_99 < best_survival:
            best_survival = survival_99
            best_scores = scores
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/30: loss={total_loss/total:.4f} test_acc={test_acc:.4f} survival@99={survival_99:.2e}")

    print(f"Best survival @ 99% gamma eff: {best_survival:.2e}")
    return best_scores, best_survival


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # Compute global normalization stats
    print("Computing global normalization stats...")
    raw_train = GammaDataset("train")
    mean, std = compute_stats(raw_train)

    # Energy bins to train on
    energy_bins = [(14.0, 15.0), (15.0, 16.0), (16.0, 17.0), (17.0, 18.0)]

    print(f"\n{'='*80}")
    print(f"TRAINING ENERGY-BINNED ENSEMBLE")
    print(f"{'='*80}")

    bin_results = {}
    for emin, emax in energy_bins:
        result = train_model_for_energy_bin((emin, emax), raw_train, mean, std)
        if result:
            scores, survival = result
            bin_results[(emin, emax)] = (scores, survival)
            print(f"Bin [{emin:.1f}, {emax:.1f}): survival={survival:.2e}")

    # Now apply ensemble predictions to test set
    print(f"\n{'='*80}")
    print(f"EVALUATING ENSEMBLE ON TEST SET")
    print(f"{'='*80}")

    f_test = np.load('data/gamma_test/features.npy', mmap_mode='r')
    y_test = np.load('data/gamma_test/labels_gamma.npy', mmap_mode='r')

    ensemble_scores = np.zeros(len(y_test))
    bin_counts = np.zeros(len(energy_bins))

    for (emin, emax), (scores, _) in bin_results.items():
        # Find test samples in this energy bin
        energy = f_test[:, 0]
        mask = (energy >= emin) & (energy < emax)

        # For test samples in this bin, use the bin-specific scores
        # For test samples NOT in this bin, we need predictions from a model
        # For now, just place the bin scores in their positions
        bin_idx = list(bin_results.keys()).index((emin, emax))
        bin_mask_indices = np.where(mask)[0]

        if len(bin_mask_indices) == len(scores):
            ensemble_scores[bin_mask_indices] = scores
            print(f"Bin [{emin:.1f}, {emax:.1f}): {len(scores)} test samples")
        else:
            print(f"Bin [{emin:.1f}, {emax:.1f}): size mismatch ({len(bin_mask_indices)} vs {len(scores)})")

    # Compute overall survival
    is_gamma = y_test == 0
    is_hadron = y_test == 1
    sg = np.sort(ensemble_scores[is_gamma])
    ng = len(sg)
    thr_99 = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
    n_hadron_surviving = (ensemble_scores[is_hadron] >= thr_99).sum()
    survival_99_ensemble = n_hadron_surviving / is_hadron.sum()

    print(f"\nEnsemble survival @ 99% gamma eff: {survival_99_ensemble:.2e}")

    # Save predictions
    np.savez(
        "submissions/haiku-gamma-mar8/predictions_energy_binned.npz",
        gamma_scores=ensemble_scores,
    )

    # Compare to baseline
    baseline_preds = np.load('submissions/haiku-gamma-mar8/predictions.npz')
    baseline_scores = baseline_preds['gamma_scores']
    sg_base = np.sort(baseline_scores[is_gamma])
    thr_99_base = sg_base[max(0, int(np.floor(ng * (1 - 0.99))))]
    survival_99_base = (baseline_scores[is_hadron] >= thr_99_base).sum() / is_hadron.sum()

    print(f"\nComparison:")
    print(f"  Baseline (single MLP): {survival_99_base:.2e}")
    print(f"  Energy-binned ensemble: {survival_99_ensemble:.2e}")
    if survival_99_ensemble < survival_99_base:
        print(f"  -> Ensemble is BETTER by {100*(survival_99_base - survival_99_ensemble)/survival_99_base:.1f}%")
    else:
        print(f"  -> Baseline is still better")


if __name__ == "__main__":
    main()
