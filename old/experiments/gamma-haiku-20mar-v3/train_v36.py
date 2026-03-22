#!/usr/bin/env python3
"""v36: ResNet with batch size 256 (larger batches, faster training)"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test, check_gpu_free
from verify import evaluate

# check_gpu_free()
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

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)  # 2x batch
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.muon_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.muon_bn1 = nn.BatchNorm2d(32)
        self.muon_res1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )
        self.muon_res2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.muon_skip2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2),
            nn.BatchNorm2d(64)
        )
        self.muon_pool = nn.AdaptiveAvgPool2d(1)
        self.electron_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.electron_bn1 = nn.BatchNorm2d(16)
        self.electron_res1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16)
        )
        self.electron_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 16 + 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, matrices, features):
        muon = matrices[:, 1:2]
        electron = matrices[:, 0:1]
        m = torch.relu(self.muon_bn1(self.muon_conv1(muon)))
        m = m + self.muon_res1(m)
        m = torch.relu(self.muon_skip2(m) + self.muon_res2(m))
        m = self.muon_pool(m).view(m.size(0), -1)
        e = torch.relu(self.electron_bn1(self.electron_conv1(electron)))
        e = e + self.electron_res1(e)
        e = self.electron_pool(e).view(e.size(0), -1)
        x = torch.cat([m, e, features], dim=1)
        return self.fusion(x)

model = ResNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=30)
weight = torch.tensor([1.0, 20.0]).cuda()
criterion = nn.CrossEntropyLoss(weight=weight)

best_val_loss = float('inf')
for epoch in range(30):
    model.train()
    train_loss = 0
    for mat, feat, label in train_loader:
        mat, feat, label = mat.cuda(), feat.cuda(), label.cuda()
        logits = model(mat, feat)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for mat, feat, label in val_loader:
            mat, feat, label = mat.cuda(), feat.cuda(), label.cuda()
            logits = model(mat, feat)
            loss = criterion(logits, label)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model_v36.pt')

    scheduler.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{30}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

model.load_state_dict(torch.load('model_v36.pt'))
model.eval()
X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).permute(0, 3, 1, 2).cuda()
f_test_tensor = torch.from_numpy(f_test.astype(np.float32)).cuda()
with torch.no_grad():
    logits = model(X_test_tensor, f_test_tensor)
    gamma_scores = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()

np.savez_compressed('predictions_v36.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v36: ResNet with batch_size=256")
print(f"Metric: {metric}")
