#!/usr/bin/env python3
"""v48: Simple CNN on flattened input (test raw Conv1D)"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test
from verify import evaluate

matrices, features, labels = load_train()
X_test, f_test, y_test = load_test()

class FlatCNNDataset(Dataset):
    def __init__(self, matrices, features, labels):
        self.flat = torch.from_numpy(matrices.reshape(matrices.shape[0], 1, -1).astype(np.float32))
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.flat[idx], self.features[idx], self.labels[idx]

n = len(labels)
n_train = int(0.8 * n)
perm = np.random.RandomState(42).permutation(n)
train_idx, val_idx = perm[:n_train], perm[n_train:]

train_ds = FlatCNNDataset(matrices[train_idx], features[train_idx], labels[train_idx])
val_ds = FlatCNNDataset(matrices[val_idx], features[val_idx], labels[val_idx])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

class FlatCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 + 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, flat, features):
        x = self.conv(flat).view(flat.size(0), -1)
        x = torch.cat([x, features], dim=1)
        return self.fc(x)

model = FlatCNN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
weight = torch.tensor([1.0, 20.0]).cuda()
criterion = nn.CrossEntropyLoss(weight=weight)

best_val_loss = float('inf')
for epoch in range(30):
    model.train()
    train_loss = 0
    for flat, feat, label in train_loader:
        flat, feat, label = flat.cuda(), feat.cuda(), label.cuda()
        logits = model(flat, feat)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for flat, feat, label in val_loader:
            flat, feat, label = flat.cuda(), feat.cuda(), label.cuda()
            logits = model(flat, feat)
            loss = criterion(logits, label)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model_v48.pt')
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1:2d}/30: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

model.load_state_dict(torch.load('model_v48.pt'))
model.eval()
X_test_flat = torch.from_numpy(X_test.reshape(X_test.shape[0], 1, -1).astype(np.float32)).cuda()
f_test_tensor = torch.from_numpy(f_test.astype(np.float32)).cuda()
with torch.no_grad():
    logits = model(X_test_flat, f_test_tensor)
    gamma_scores = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()

np.savez_compressed('predictions_v48.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v48: Conv1D on flattened input")
print(f"Metric: {metric}")
