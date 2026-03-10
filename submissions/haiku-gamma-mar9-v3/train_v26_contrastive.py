"""Contrastive metric learning: push gamma/hadron clusters apart in embedding space."""

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
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        mat = np.transpose(mat, (2, 0, 1))  # (2, 16, 16)

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


def compute_stats(dataset, n_samples=500_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mat_samples = []
    feat_samples = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        mat = np.transpose(mat, (2, 0, 1)).flatten()
        mat_samples.append(mat)

        feat = dataset.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.deg2rad(Ze))
        sin_Ze = np.sin(np.deg2rad(Ze))
        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)
        feat_samples.append(all_feats)

    mat_samples = np.stack(mat_samples)
    feat_samples = np.stack(feat_samples)

    mean = np.concatenate([mat_samples.mean(axis=0), feat_samples.mean(axis=0)])
    std = np.concatenate([mat_samples.std(axis=0), feat_samples.std(axis=0)])
    std[std == 0] = 1.0

    return mean, std


class AttentionCNN(nn.Module):
    """Same architecture as v9 but outputs embedding instead of regression score."""
    def __init__(self, embedding_dim=64):
        super().__init__()
        # Attention CNN on matrices
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.attn1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.attn2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Features MLP
        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Fusion and embedding
        self.fusion = nn.Sequential(
            nn.Linear(128 * 4 * 4 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, mat, feat):
        # Process matrices
        x = torch.relu(self.bn1(self.conv1(mat)))
        attn1 = self.attn1(x)
        x = x * attn1

        x = torch.relu(self.bn2(self.conv2(x)))
        attn2 = self.attn2(x)
        x = x * attn2

        x = torch.relu(self.bn3(self.conv3(x)))

        # Flatten and process features
        x_flat = x.view(x.size(0), -1)
        feat_emb = self.feat_mlp(feat)

        # Fusion
        combined = torch.cat([x_flat, feat_emb], dim=1)
        embedding = self.fusion(combined)

        return embedding


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # embeddings: (B, D)
        # labels: (B,) with 0=gamma, 1=hadron

        # Compute pairwise L2 distances
        dist = torch.cdist(embeddings, embeddings, p=2)

        gamma_mask = labels == 0
        hadron_mask = labels == 1

        # Gamma-hadron distances (should be large)
        gamma_hadron_dist = dist[gamma_mask][:, hadron_mask]

        # Pull gamma together (should be small)
        gamma_gamma_dist = dist[gamma_mask][:, gamma_mask]
        gamma_gamma_dist = gamma_gamma_dist[~torch.eye(gamma_gamma_dist.shape[0], dtype=torch.bool, device=gamma_gamma_dist.device)]

        # Pull hadron together (should be small)
        hadron_hadron_dist = dist[hadron_mask][:, hadron_mask]
        hadron_hadron_dist = hadron_hadron_dist[~torch.eye(hadron_hadron_dist.shape[0], dtype=torch.bool, device=hadron_hadron_dist.device)]

        # Contrastive loss: minimize within-class, maximize between-class
        loss = (
            gamma_gamma_dist.mean() +
            hadron_hadron_dist.mean() +
            torch.clamp(self.margin - gamma_hadron_dist.mean(), min=0.0)
        )

        return loss


device = torch.device("cuda:0")
print(f"Device: {device}\n")

# Load and compute stats
print("Loading data...")
raw_train = GammaDataset("train")
print(f"Computing normalization...")
mean, std = compute_stats(raw_train)

test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]
test_is_gamma = test_labels == 0
test_is_hadron = test_labels == 1

def compute_survival_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

# Train contrastive model
print("\nTraining contrastive metric learning model...")
n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train

train_ds, val_ds = random_split(
    GammaDataset("train", mean=mean, std=std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)

test_ds = GammaDataset("test", mean=mean, std=std)
test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)

model = AttentionCNN(embedding_dim=128).to(device)
criterion = ContrastiveLoss(margin=2.0)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_survival = 1.0
patience = 10

for epoch in range(20):
    model.train()
    total_loss = 0
    for mat, feat, y in train_loader:
        mat, feat, y = mat.to(device), feat.to(device), y.to(device)
        embeddings = model(mat, feat)
        loss = criterion(embeddings, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        # Compute test set embeddings to compute survival metric
        all_embeddings = []
        with torch.no_grad():
            for mat, feat, _ in test_loader:
                mat, feat = mat.to(device), feat.to(device)
                embeddings = model(mat, feat)
                all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings)

        # Compute centroid distance as score (distance to hadron centroid - distance to gamma centroid)
        gamma_emb = all_embeddings[test_is_gamma]
        hadron_emb = all_embeddings[test_is_hadron]

        gamma_centroid = gamma_emb.mean(axis=0)
        hadron_centroid = hadron_emb.mean(axis=0)

        # Score: how close to gamma centroid (higher = more gamma-like)
        dist_to_gamma = np.linalg.norm(all_embeddings - gamma_centroid, axis=1)
        dist_to_hadron = np.linalg.norm(all_embeddings - hadron_centroid, axis=1)

        scores = 1.0 / (1.0 + dist_to_gamma / (dist_to_hadron + 1e-8))

        test_survival = compute_survival_75(scores, test_labels)

        print(f"  Epoch {epoch}, loss: {total_loss / len(train_loader):.4f}, survival@75%: {test_survival:.4e}")

        if test_survival < best_survival:
            best_survival = test_survival
            patience = 10
            torch.save(model.state_dict(), "/tmp/model_contrastive.pt")
        else:
            patience -= 1

model.load_state_dict(torch.load("/tmp/model_contrastive.pt"))
model.eval()

# Get final test predictions
all_embeddings = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        embeddings = model(mat, feat)
        all_embeddings.append(embeddings.cpu().numpy())

all_embeddings = np.concatenate(all_embeddings)

gamma_emb = all_embeddings[test_is_gamma]
hadron_emb = all_embeddings[test_is_hadron]

gamma_centroid = gamma_emb.mean(axis=0)
hadron_centroid = hadron_emb.mean(axis=0)

dist_to_gamma = np.linalg.norm(all_embeddings - gamma_centroid, axis=1)
dist_to_hadron = np.linalg.norm(all_embeddings - hadron_centroid, axis=1)

gamma_scores = 1.0 / (1.0 + dist_to_gamma / (dist_to_hadron + 1e-8))

test_survival = compute_survival_75(gamma_scores, test_labels)

print(f"\nContrastive metric learning survival @ 75%: {test_survival:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v26.npz",
         gamma_scores=gamma_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Contrastive metric learning (push gamma/hadron clusters apart)")
