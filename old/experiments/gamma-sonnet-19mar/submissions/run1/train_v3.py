"""
v3: Deeper CNN with attention mechanism on muon channel + better training.
Key insights from v2: CNN+MLP hybrid achieves 4.09e-04 (beat MLP 10x).
The spatial muon pattern is KEY - gamma showers have near-zero muons.
Improvements:
- Deeper CNN with residual connections
- Attention on the muon channel specifically
- Focal loss to handle class imbalance better
- More epochs with better LR schedule
- Separate processing of muon vs electron channels
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def engineer_features(f):
    E = f[:, 0]
    Ze = f[:, 1]
    Az = f[:, 2]
    Ne = f[:, 3]
    Nmu = f[:, 4]

    ne_nmu_diff = Ne - Nmu
    Ze_norm = Ze / 30.0
    Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7
    E_norm = (E - 16.0) / 1.0

    Az_rad = np.radians(Az)
    Az_cos = np.cos(Az_rad)
    Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze))
    ne_e_ratio = Ne - E

    feats = np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff,
        cos_ze,
        ne_e_ratio,
        Ne * Ze_norm,
        Nmu * cos_ze,
    ], axis=1).astype(np.float32)
    return feats


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean=None, feat_std=None):
        self.matrices = matrices
        self.features = features
        self.labels = labels
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        # log1p transform for sparse matrices
        mat = np.log1p(mat)
        mat = mat.transpose(2, 0, 1)  # (2, 16, 16)

        feat = self.features[idx].copy()
        if self.feat_mean is not None:
            feat = (feat - self.feat_mean) / self.feat_std

        return (torch.FloatTensor(mat),
                torch.FloatTensor(feat),
                int(self.labels[idx]))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class SpatialAttention(nn.Module):
    """Channel attention focusing on muon vs electron channels."""
    def __init__(self, ch):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // 4),
            nn.ReLU(),
            nn.Linear(ch // 4, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.att(x).unsqueeze(-1).unsqueeze(-1)
        return x * att


class GammaCNNv3(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        # Process electron and muon channels separately first
        self.stem_em = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stem_mu = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Main CNN backbone
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.MaxPool2d(2),  # 8x8
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            SpatialAttention(128),
            nn.MaxPool2d(2),  # 4x4
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),
            SpatialAttention(256),
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        cnn_out = 256

        # Feature MLP
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(cnn_out + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, mat, feat):
        em = self.stem_em(mat[:, 0:1])  # electron channel
        mu = self.stem_mu(mat[:, 1:2])  # muon channel
        x = torch.cat([em, mu], dim=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        cnn_out = self.pool(x)

        feat_out = self.feat_mlp(feat)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        return self.head(combined)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance: down-weights easy negatives."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def survival_at_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])[::-1]
    ng = len(sg)
    idx = int(np.ceil(0.75 * ng)) - 1
    thr = sg[min(idx, ng - 1)]
    survival = (scores[is_hadron] >= thr).sum() / is_hadron.sum()
    return survival


def main():
    print("Loading data...")
    f_train_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_train_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_train_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')

    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test_raw = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r')

    f_train = engineer_features(np.array(f_train_raw))
    f_test = engineer_features(np.array(f_test_raw))
    y_train = np.array(y_train_raw)
    y_test = np.array(y_test_raw)

    feat_mean = f_train.mean(0)
    feat_std = f_train.std(0)
    feat_std[feat_std < 1e-8] = 1.0

    # Train/val split (90/10)
    n = len(f_train)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    n_val = int(n * 0.1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    print(f"Train: {len(tr_idx):,}, Val: {len(val_idx):,}")

    train_ds = GammaDataset(m_train_raw[tr_idx], f_train[tr_idx], y_train[tr_idx], feat_mean, feat_std)
    val_ds = GammaDataset(m_train_raw[val_idx], f_train[val_idx], y_train[val_idx], feat_mean, feat_std)
    test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNNv3(f_train.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    n_gamma = (y_train[tr_idx] == 0).sum()
    n_hadron = (y_train[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    print(f"Class weight gamma: {w_gamma:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, epochs=25, steps_per_epoch=len(train_loader)
    )
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    best_val_survival = 1.0
    best_scores_test = None
    n_epochs = 25

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_total = 0
        for mat, feat, y in train_loader:
            mat, feat, y = mat.to(DEVICE), feat.to(DEVICE), y.to(DEVICE)
            logits = model(mat, feat)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)
            n_total += len(y)

        # Validate
        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for mat, feat, y in val_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                logits = model(mat, feat)
                probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
                val_scores.append(probs)
                val_labels.append(y.numpy())
        val_scores = np.concatenate(val_scores)
        val_labels = np.concatenate(val_labels)
        val_survival = survival_at_75(val_scores, val_labels)

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:2d}/{n_epochs}: loss={total_loss/n_total:.4f} "
              f"val_survival@75={val_survival:.2e} lr={lr:.5f}")

        if val_survival < best_val_survival:
            best_val_survival = val_survival
            test_scores = []
            with torch.no_grad():
                for mat, feat, y in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    logits = model(mat, feat)
                    probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
                    test_scores.append(probs)
            best_scores_test = np.concatenate(test_scores)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v3.pt")
            print(f"  -> New best: {val_survival:.2e}")

    print(f"\nBest val survival@75: {best_val_survival:.2e}")
    np.save(f"{OUT_DIR}/probs_v3.npy", best_scores_test)
    np.savez(f"{OUT_DIR}/predictions_v3.npz", gamma_scores=best_scores_test)
    np.savez(f"{OUT_DIR}/predictions.npz", gamma_scores=best_scores_test)
    print(f"Saved predictions ({len(best_scores_test)} scores)")

    print("\n---")
    print(f"metric: {best_val_survival:.4e}")
    print("description: Deep ResNet CNN + attention + focal loss, separate em/mu processing, 25 epochs")


if __name__ == "__main__":
    main()
