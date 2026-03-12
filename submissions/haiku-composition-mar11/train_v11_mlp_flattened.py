"""v11: Pure MLP on flattened matrices + features.

No convolution - just treat 16×16×2 matrix as 512D vector.
See if CNNs are actually necessary or if MLP works just as well.
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
        mat = np.log1p(mat)
        mat_flat = mat.reshape(-1)  # Flatten to 512D

        feat = self.features[idx].astype(np.float32)  # (5,)

        if self.mat_mean is not None:
            mat_flat = (mat_flat - self.mat_mean) / (self.mat_std + 1e-8)

        E, Ze, Az, Ne, Nmu = feat
        eng = np.array([E, Ze, Az, Ne, Nmu,
                       Ne - Nmu,
                       np.cos(np.radians(Ze)),
                       np.sin(np.radians(Ze))], dtype=np.float32)

        if self.feat_mean is not None:
            eng = (eng - self.feat_mean) / (self.feat_std + 1e-8)

        combined = np.concatenate([mat_flat, eng])
        label = int(self.labels[idx])
        return torch.from_numpy(combined), label


def compute_stats(dataset, n_samples=100_000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mat_samples = []
    feat_samples = []
    for idx in indices:
        mat = dataset.matrices[idx].astype(np.float32)
        mat = np.log1p(mat)
        mat_flat = mat.reshape(-1)
        feat = dataset.features[idx].astype(np.float32)

        E, Ze, Az, Ne, Nmu = feat
        eng = np.array([E, Ze, Az, Ne, Nmu, Ne - Nmu,
                       np.cos(np.radians(Ze)),
                       np.sin(np.radians(Ze))], dtype=np.float32)

        mat_samples.append(mat_flat)
        feat_samples.append(eng)

    mat_samples = np.stack(mat_samples)
    feat_samples = np.stack(feat_samples)

    mat_mean = mat_samples.mean(axis=0)
    mat_std = mat_samples.std(axis=0)
    mat_std[mat_std == 0] = 1.0

    feat_mean = feat_samples.mean(axis=0)
    feat_std = feat_samples.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    return mat_mean, mat_std, feat_mean, feat_std


class PureMLPComposition(nn.Module):
    def __init__(self):
        super().__init__()
        # 512D flattened matrix + 8 features = 520D input
        self.net = nn.Sequential(
            nn.Linear(520, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda:0")
print(f"Device: {device}\n")

print("Loading data and computing statistics...")
raw_train = CompositionDataset("train")
mat_mean, mat_std, feat_mean, feat_std = compute_stats(raw_train)

test_ds = CompositionDataset("test", mat_mean, mat_std, feat_mean, feat_std)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
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

train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True,
                         num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

print("Training v11: Pure MLP on flattened matrices (520D input)...")
model = PureMLPComposition().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2e-3, steps_per_epoch=len(train_loader), epochs=30
)

best_val_acc = 0
patience = 10

for epoch in range(30):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    if epoch % 5 == 0:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.numpy())

        val_preds = np.concatenate(all_preds)
        val_labels = np.concatenate(all_labels)
        val_acc = (val_preds == val_labels).mean()

        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 10
            torch.save(model.state_dict(), "/tmp/model_composition_v11.pt")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

print("Running inference...")
model.load_state_dict(torch.load("/tmp/model_composition_v11.pt"))
model.eval()

all_preds = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)

test_preds = np.concatenate(all_preds)
test_acc = (test_preds == test_labels).mean()

np.savez("submissions/haiku-composition-mar11/predictions_v11.npz",
         predictions=test_preds)

print(f"\n---")
print(f"metric: {test_acc:.4f}")
print(f"description: Pure MLP (1024→512→256→128) on 520D input (512D matrix + 8 features)")
