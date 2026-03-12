"""v10: Deeper ResNet variant (3 residual blocks instead of 2).

Try expanding v4's successful ResNet approach.
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
        mat = self.matrices[idx].astype(np.float32)

        if self.mat_mean is not None:
            mat = (mat - self.mat_mean) / (self.mat_std + 1e-8)

        E, Ze, Az, Ne, Nmu = self.features[idx].astype(np.float32)

        all_feats = np.array([E, Ze, Az, Ne, Nmu,
                             Ne - Nmu,
                             np.cos(np.radians(Ze)),
                             np.sin(np.radians(Ze))], dtype=np.float32)

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

        E, Ze, Az, Ne, Nmu = feat
        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne - Nmu,
                             np.cos(np.radians(Ze)),
                             np.sin(np.radians(Ze))], dtype=np.float32)

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetDeeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mat_conv0 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 3 residual blocks (instead of 2)
        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        self.res_block3 = ResidualBlock(128, 256, stride=1)

        self.mat_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, mat, feat):
        x = self.mat_conv0(mat)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.mat_pool(x).view(x.size(0), -1)

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

print("Training v10: Deeper ResNet (3 blocks) + 8 features...")
model = ResNetDeeper().to(device)
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

        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 10
            torch.save(model.state_dict(), "/tmp/model_composition_v10.pt")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

print("Running inference...")
model.load_state_dict(torch.load("/tmp/model_composition_v10.pt"))
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

np.savez("submissions/haiku-composition-mar11/predictions_v10.npz",
         predictions=test_preds)

print(f"\n---")
print(f"metric: {test_acc:.4f}")
print(f"description: Deeper ResNet (3 blocks, 256 channels) + 8 features")
