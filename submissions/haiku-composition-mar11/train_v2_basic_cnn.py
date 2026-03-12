"""v2: Basic CNN without attention (testing if attention hurts on composition).

Simpler architecture: 2-layer CNN without attention blocks.
Keep engineered features, but simplify spatial pathway.
Classes: 0=proton, 1=helium, 2=carbon, 3=silicon, 4=iron
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim


class CompositionDataset(Dataset):
    def __init__(self, split: str, mat_mean=None, mat_std=None, feat_mean=None, feat_std=None):
        self.matrices = np.load(f"data/composition_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/composition_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/composition_{split}/labels_composition.npy", mmap_mode="r")
        self.mat_mean = mat_mean
        self.mat_std = mat_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        feat = self.features[idx].astype(np.float32)  # (5,): E, Ze, Az, Ne, Nmu

        if self.mat_mean is not None:
            mat = (mat - self.mat_mean) / (self.mat_std + 1e-8)

        # Engineer features
        E = feat[0]
        Ze = feat[1]
        Az = feat[2]
        Ne = feat[3]
        Nmu = feat[4]

        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.radians(Ze * 180 / np.pi))
        sin_Ze = np.sin(np.radians(Ze * 180 / np.pi))

        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        if self.feat_mean is not None:
            all_feats = (all_feats - self.feat_mean) / (self.feat_std + 1e-8)

        mat = np.transpose(mat, (2, 0, 1))

        label = int(self.labels[idx])
        return torch.from_numpy(mat), torch.from_numpy(all_feats), label


def compute_stats(dataset, n_samples=100_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mat_samples = []
    feat_samples = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        feat = dataset.features[idx].astype(np.float32)

        E = feat[0]
        Ze = feat[1]
        Az = feat[2]
        Ne = feat[3]
        Nmu = feat[4]

        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.radians(Ze * 180 / np.pi))
        sin_Ze = np.sin(np.radians(Ze * 180 / np.pi))

        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        mat_samples.append(mat)
        feat_samples.append(all_feats)

    mat_samples = np.stack(mat_samples)
    feat_samples = np.stack(feat_samples)

    mat_mean = mat_samples.mean(axis=0)
    mat_std = mat_samples.std(axis=0)
    mat_std[mat_std == 0] = 1.0

    feat_mean = feat_samples.mean(axis=0)
    feat_std = feat_samples.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    return mat_mean, mat_std, feat_mean, feat_std


class BasicCNNComposition(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN without attention
        self.mat_cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Feature pathway: 8 engineered features
        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Fusion and final layers (5 classes)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, mat, feat):
        x = self.mat_cnn(mat).view(mat.size(0), -1)
        f = self.feat_mlp(feat)
        fused = torch.cat([x, f], dim=1)
        out = self.fusion(fused)
        return out


device = torch.device("cuda:0")
print(f"Device: {device}\n")

print("Loading data and computing statistics...")
raw_train = CompositionDataset("train")
mat_mean, mat_std, feat_mean, feat_std = compute_stats(raw_train)

test_ds = CompositionDataset("test", mat_mean, mat_std, feat_mean, feat_std)
test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
test_labels = np.load("data/composition_test/labels_composition.npy")[:]

print("Preparing train/val split...")
np.random.seed(42)
torch.manual_seed(42)

n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train
train_ds, val_ds = random_split(
    CompositionDataset("train", mat_mean, mat_std, feat_mean, feat_std),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False,
                       num_workers=4, pin_memory=True)

print("Training v2: Basic CNN (no attention) + engineered features...")
model = BasicCNNComposition().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_val_acc = 0
patience = 10

for epoch in range(30):
    model.train()
    total_loss = 0
    for mat, feat, y in train_loader:
        mat, feat = mat.to(device), feat.to(device)
        y = y.to(device)
        logits = model(mat, feat)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for mat, feat, y in val_loader:
                mat, feat = mat.to(device), feat.to(device)
                logits = model(mat, feat)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.numpy())

        val_preds = np.concatenate(all_preds)
        val_labels = np.concatenate(all_labels)
        val_acc = (val_preds == val_labels).mean()

        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 10
            torch.save(model.state_dict(), "/tmp/model_composition_v2.pt")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

print("Running inference...")
model.load_state_dict(torch.load("/tmp/model_composition_v2.pt"))
model.eval()

all_preds = []
with torch.no_grad():
    for mat, feat, _ in test_loader:
        mat, feat = mat.to(device), feat.to(device)
        logits = model(mat, feat)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)

test_preds = np.concatenate(all_preds)
test_acc = (test_preds == test_labels).mean()

np.savez("submissions/haiku-composition-mar11/predictions_v2.npz",
         predictions=test_preds)

print(f"\n---")
print(f"metric: {test_acc:.4f}")
print(f"description: Basic CNN (no attention) + 8 engineered features")
