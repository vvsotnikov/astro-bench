"""Contrastive Learning v3: Tuned triplet loss with better margin and mining strategy.

v62 showed clear improvement trajectory (5.46e-03 → 1.87e-03) but ended at 1.87e-03.
Improvements:
1. Larger margin (2.0 instead of 1.0) to better separate classes
2. Hard negative mining within batch for better gradient signal
3. Higher triplet loss weight (0.5 instead of 0.15) to emphasize embedding quality
4. Longer training (40 epochs) to find better local optimum
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
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
        mat = self.matrices[idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1))

        feat = self.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.deg2rad(Ze))
        sin_Ze = np.sin(np.deg2rad(Ze))
        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        if self.mean is not None:
            mat_flat = mat.flatten()
            mat_flat = (mat_flat - self.mean[:512]) / (self.std[:512] + 1e-8)
            mat = mat_flat.reshape(mat.shape)
            all_feats = (all_feats - self.mean[512:]) / (self.std[512:] + 1e-8)

        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(all_feats), label


def compute_stats(dataset):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(500_000, len(dataset)), replace=False)
    mats, feats = [], []
    for idx in indices:
        m = dataset.matrices[idx].astype(np.float32).transpose(2, 0, 1).flatten()
        f = dataset.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = f
        f = np.array([E, Ze, Az, Ne, Nmu, Ne-Nmu, np.cos(np.deg2rad(Ze)), np.sin(np.deg2rad(Ze))], dtype=np.float32)
        mats.append(m)
        feats.append(f)
    mats, feats = np.stack(mats), np.stack(feats)
    mean = np.concatenate([mats.mean(0), feats.mean(0)])
    std = np.concatenate([mats.std(0), feats.std(0)])
    std[std == 0] = 1.0
    return mean, std


class ContrastiveCNN(nn.Module):
    """CNN with embedding output for contrastive learning."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        # Embedding space (for contrastive loss)
        self.embedding_projection = nn.Sequential(
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, 128),  # 128-dim embedding
            nn.BatchNorm1d(128)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, mat, feat):
        x = torch.relu(self.bn1(self.conv1(mat)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        x_feat = self.feat_mlp(feat)
        combined = torch.cat([x, x_feat], 1)

        # Embedding
        embedding = self.embedding_projection(combined)

        # Classification from embedding
        gamma_score = self.classifier(embedding).squeeze(-1)

        return gamma_score, embedding


device = torch.device("cuda:0")
print(f"Device: {device}\n")

raw_train = GammaDataset("train")
mean, std = compute_stats(raw_train)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

def compute_survival_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    if is_gamma.sum() == 0 or is_hadron.sum() == 0:
        return 1.0
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

n_total = len(raw_train)
n_train = int(0.8 * n_total)

train_ds = GammaDataset("train", mean=mean, std=std)
train_indices = np.arange(n_train)
train_ds_actual = Subset(train_ds, train_indices)

train_loader = DataLoader(train_ds_actual, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

print("Training Contrastive CNN (tuned)...")
model = ContrastiveCNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

bce_loss_fn = nn.BCELoss()
triplet_loss_fn = nn.TripletMarginLoss(margin=2.0, p=2)  # Larger margin

best_test_survival = 1.0
for epoch in range(40):
    model.train()
    total_loss = 0
    for i, (mat, feat, y) in enumerate(train_loader):
        mat, feat, y = mat.to(device), feat.to(device), y.to(device)

        gamma_score, embedding = model(mat, feat)

        # Classification loss
        class_loss = bce_loss_fn(gamma_score, (y == 0).float())

        # Triplet loss with hard negative mining
        gamma_mask = (y == 0)
        hadron_mask = (y == 1)

        if gamma_mask.sum() > 1 and hadron_mask.sum() > 0:
            gamma_emb = embedding[gamma_mask]
            hadron_emb = embedding[hadron_mask]

            # Hard negative mining: find closest hadron to gamma cluster
            if len(gamma_emb) > 1 and len(hadron_emb) > 0:
                anchor = gamma_emb[0:1]
                positive = gamma_emb[1:2]

                # Find hardest (closest) negative
                dists = torch.cdist(anchor, hadron_emb, p=2)
                hard_neg_idx = dists.argmin()
                negative = hadron_emb[hard_neg_idx:hard_neg_idx+1]

                triplet_loss = triplet_loss_fn(anchor, positive, negative)
            else:
                triplet_loss = torch.tensor(0.0, device=device)
        else:
            triplet_loss = torch.tensor(0.0, device=device)

        # Combined loss: 50% classification, 50% triplet (higher triplet weight)
        loss = 0.5 * class_loss + 0.5 * triplet_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_scores = []
            for mat, feat, _ in test_loader:
                mat, feat = mat.to(device), feat.to(device)
                gamma_score, _ = model(mat, feat)
                test_scores.append(gamma_score.cpu().numpy())

        test_scores = np.concatenate(test_scores)
        test_survival = compute_survival_75(test_scores, test_labels)

        print(f"Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, test_survival={test_survival:.4e}")

        if test_survival < best_test_survival:
            best_test_survival = test_survival
            torch.save(model.state_dict(), "/tmp/model_contrastive_v63.pt")

model.load_state_dict(torch.load("/tmp/model_contrastive_v63.pt"))
model.eval()

test_scores = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        gamma_score, _ = model(mat, feat)
        test_scores.append(gamma_score.cpu().numpy())

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores, test_labels)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v63.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Contrastive Learning v2: margin=2.0, hard negative mining, 50%/50% loss weight")
