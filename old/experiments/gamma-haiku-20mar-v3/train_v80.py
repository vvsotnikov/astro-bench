#!/usr/bin/env python3
"""v80: Simpler MLP on aggregated spatial features - no CNN, just pool early"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test
from verify import evaluate

torch.manual_seed(42)
np.random.seed(42)

matrices, features, labels = load_train()
X_test, f_test, y_test = load_test()

class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels):
        self.matrices = torch.from_numpy(matrices.astype(np.float32)).permute(0, 3, 1, 2)
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.matrices[idx], self.features[idx], self.labels[idx]

n = len(labels)
n_train = int(0.8 * n)
perm = np.random.RandomState(42).permutation(n)
train_idx, val_idx = perm[:n_train], perm[n_train:]

train_ds = GammaDataset(matrices[train_idx], features[train_idx], labels[train_idx])
val_ds = GammaDataset(matrices[val_idx], features[val_idx], labels[val_idx])
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal processing: flatten spatial data immediately
        # 16*16*2 = 512 spatial features + 5 scalar features = 517 total
        # Aggregate spatial features to reduce dimensionality
        self.spatial_mlp = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Fusion of aggregated spatial + scalar features
        self.fusion = nn.Sequential(
            nn.Linear(64 + 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, matrices, features):
        # Flatten spatial dimensions: (B, 2, 16, 16) -> (B, 512)
        spatial = matrices.view(matrices.size(0), -1)
        # Process spatial features
        spatial_feat = self.spatial_mlp(spatial)
        # Fusion with scalar features
        x = torch.cat([spatial_feat, features], dim=1)
        return self.fusion(x)

model = SimpleMLP().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=30)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 20.0]).cuda())
best_val_loss = float('inf')

for epoch in range(30):
    model.train()
    train_loss = 0
    for mat, feat, label in train_loader:
        mat, feat, label = mat.cuda(), feat.cuda(), label.cuda()
        loss = criterion(model(mat, feat), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for mat, feat, label in val_loader:
            mat, feat, label = mat.cuda(), feat.cuda(), label.cuda()
            val_loss += criterion(model(mat, feat), label).item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model_v80.pt')

    scheduler.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{30}: train={train_loss:.4f} val={val_loss:.4f}")

model.load_state_dict(torch.load('model_v80.pt'))
model.eval()

all_scores = []
for i in range(0, len(X_test), 1024):
    end_i = min(i + 1024, len(X_test))
    X_b = torch.from_numpy(X_test[i:end_i].astype(np.float32)).permute(0, 3, 1, 2).cuda()
    f_b = torch.from_numpy(f_test[i:end_i].astype(np.float32)).cuda()
    with torch.no_grad():
        scores = torch.softmax(model(X_b, f_b), dim=1)[:, 0].cpu().numpy()
    all_scores.append(scores)

gamma_scores = np.concatenate(all_scores)
np.savez_compressed('predictions_v80.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v80: Simple MLP (no CNN, spatial flattening)")
print(f"Metric: {metric}")
