"""v3: Larger CNN + engineered features.

Add Ne/Nmu ratio and log-transformed features to help the model leverage
the physics-relevant features more directly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.matrices = np.load(f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        feat = self.features[idx].astype(np.float32)  # (5,) [E, Ze, Az, Ne, Nmu]

        # Engineer features
        E, Ze, Az, Ne, Nmu = feat
        ne_nmu_ratio = np.log(Ne / (Nmu + 1e-6) + 1e-6)  # log ratio
        ne_nmu_product = Ne * Nmu
        eng_feat = np.array([ne_nmu_ratio, ne_nmu_product, np.log(Ne), np.log(Nmu)], dtype=np.float32)

        if self.mean is not None:
            feat_norm = (feat - self.mean[-5:]) / self.std[-5:]
            eng_norm = (eng_feat - self.mean[-9:-5]) / self.std[-9:-5]
            feat_norm = np.concatenate([feat_norm, eng_norm])
            mat_flat = mat.flatten()
            mat_norm = (mat_flat - self.mean[:-9]) / self.std[:-9]
            return torch.from_numpy(mat.transpose(2, 0, 1)), torch.from_numpy(feat_norm), int(self.labels[idx])

        feat_final = np.concatenate([feat, eng_feat])
        return torch.from_numpy(mat.transpose(2, 0, 1)), torch.from_numpy(feat_final), int(self.labels[idx])

    def get_stats(self, n_samples=500_000):
        """Pre-compute statistics for normalization."""
        rng = np.random.default_rng(42)
        indices = rng.choice(len(self.labels), size=min(n_samples, len(self.labels)), replace=False)
        mat_samples = []
        feat_samples = []
        for idx in indices:
            mat = self.matrices[idx].flatten().astype(np.float32)
            feat = self.features[idx].astype(np.float32)
            E, Ze, Az, Ne, Nmu = feat
            ne_nmu_ratio = np.log(Ne / (Nmu + 1e-6) + 1e-6)
            ne_nmu_product = Ne * Nmu
            eng_feat = np.array([ne_nmu_ratio, ne_nmu_product, np.log(Ne), np.log(Nmu)], dtype=np.float32)
            mat_samples.append(mat)
            feat_samples.append(np.concatenate([feat, eng_feat]))
        mat_arr = np.stack(mat_samples)
        feat_arr = np.stack(feat_samples)
        mat_mean = mat_arr.mean(axis=0)
        mat_std = mat_arr.std(axis=0)
        mat_std[mat_std == 0] = 1.0
        feat_mean = feat_arr.mean(axis=0)
        feat_std = feat_arr.std(axis=0)
        feat_std[feat_std == 0] = 1.0
        return np.concatenate([mat_mean, feat_mean]), np.concatenate([mat_std, feat_std])


class CNNFeatureExtractor(nn.Module):
    """Larger CNN for 16x16x2 spatial matrices."""
    def __init__(self, out_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x


class HybridModel(nn.Module):
    def __init__(self, feature_dim=9, cnn_out=96, hidden=256):
        super().__init__()
        self.cnn = CNNFeatureExtractor(out_dim=cnn_out)
        self.feat_proj = nn.Linear(feature_dim, cnn_out)
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + cnn_out, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, 2),
        )

    def forward(self, mat, feat):
        cnn_out = self.cnn(mat)
        feat_out = self.feat_proj(feat)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        return self.fusion(combined)


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Loading data and computing stats...")
    raw_train = GammaDataset("train")
    mean, std = raw_train.get_stats()
    print(f"  mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  std range:  [{std.min():.2f}, {std.max():.2f}]")

    train_ds = GammaDataset("train", mean=mean, std=std)
    train_loader = DataLoader(
        train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True
    )

    test_ds = GammaDataset("test", mean=mean, std=std)
    test_loader = DataLoader(
        test_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True
    )

    model = HybridModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Class weights
    labels_all = raw_train.labels[:]
    n_gamma = (labels_all == 0).sum()
    n_hadron = (labels_all == 1).sum()
    w_gamma = len(labels_all) / (2 * n_gamma)
    w_hadron = len(labels_all) / (2 * n_hadron)
    class_weights = torch.tensor([w_gamma, w_hadron], dtype=torch.float32).to(device)
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    n_epochs = 40
    best_survival = 1.0
    for epoch in range(n_epochs):
        model.train()
        correct = total = 0
        total_loss = 0
        for mat, feat, y in train_loader:
            mat, feat, y = mat.to(device), feat.to(device), y.to(device)
            logits = model(mat, feat)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / total
        lr = optimizer.param_groups[0]["lr"]

        # Evaluate
        model.eval()
        test_correct = test_total = 0
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for mat, feat, y in test_loader:
                mat, feat, y = mat.to(device), feat.to(device), y.to(device)
                logits = model(mat, feat)
                probs = torch.softmax(logits, dim=1)
                gamma_scores = probs[:, 0]
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += len(y)
                all_scores.append(gamma_scores.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        test_acc = test_correct / test_total
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        is_gamma = labels == 0
        is_hadron = labels == 1
        sg = np.sort(scores[is_gamma])
        ng = len(sg)
        thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
        n_hadron_surviving = (scores[is_hadron] >= thr_75).sum()
        survival_75 = n_hadron_surviving / is_hadron.sum()

        print(
            f"Epoch {epoch+1:2d}/{n_epochs}: "
            f"train_loss={total_loss/total:.4f} train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} survival@75={survival_75:.2e} lr={lr:.6f}"
        )

        if survival_75 < best_survival:
            best_survival = survival_75
            best_scores = scores
            torch.save(model.state_dict(), "model_v3.pt")

    print(f"\nBest survival @ 75% gamma eff: {best_survival:.2e}")
    np.savez("predictions_v3.npz", gamma_scores=best_scores)
    print(f"Saved ({len(best_scores)} scores)")


if __name__ == "__main__":
    main()
