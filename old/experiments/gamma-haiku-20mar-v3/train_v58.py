#!/usr/bin/env python3
"""v58: Simple Vision Transformer (patch-based attention on detector grids)"""
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

class SimpleViT(nn.Module):
    def __init__(self, dim=128, heads=4, depth=2):
        super().__init__()
        self.patch_embed_muon = nn.Sequential(nn.Conv2d(1, dim, kernel_size=4, stride=4), nn.Flatten(2))
        self.patch_embed_electron = nn.Sequential(nn.Conv2d(1, dim, kernel_size=4, stride=4), nn.Flatten(2))
        self.pos_embed_muon = nn.Parameter(torch.randn(1, 16+1, dim))
        self.pos_embed_electron = nn.Parameter(torch.randn(1, 16+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=256, batch_first=True)
        self.muon_transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.electron_transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.fusion = nn.Sequential(nn.Linear(2*dim+5, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, matrices, features):
        muon = matrices[:, 1:2]
        electron = matrices[:, 0:1]

        m = self.patch_embed_muon(muon)  # (B, dim, 16)
        m = m.transpose(1, 2)  # (B, 16, dim)
        m = torch.cat([self.cls_token.expand(m.size(0), -1, -1), m], dim=1)
        m = m + self.pos_embed_muon
        m = self.muon_transformer(m)
        m = m[:, 0, :]  # CLS token

        e = self.patch_embed_electron(electron)
        e = e.transpose(1, 2)
        e = torch.cat([self.cls_token.expand(e.size(0), -1, -1), e], dim=1)
        e = e + self.pos_embed_electron
        e = self.electron_transformer(e)
        e = e[:, 0, :]

        x = torch.cat([m, e, features], dim=1)
        return self.fusion(x)

model = SimpleViT().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
weight = torch.tensor([1.0, 20.0]).cuda()
criterion = nn.CrossEntropyLoss(weight=weight)
best_val_loss = float('inf')
patience_counter = 0

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
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for mat, feat, label in val_loader:
            mat, feat, label = mat.cuda(), feat.cuda(), label.cuda()
            logits = model(mat, feat)
            loss = criterion(logits, label)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: {train_loss:.4f}/{val_loss:.4f}")
    if patience_counter >= 5:
        break

model.eval()
X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).permute(0,3,1,2).cuda()
f_test_tensor = torch.from_numpy(f_test.astype(np.float32)).cuda()
with torch.no_grad():
    logits = model(X_test_tensor, f_test_tensor)
    gamma_scores = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
np.savez_compressed('predictions_v58.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v58: Simple Vision Transformer (4 heads, 2 layers)")
print(f"Metric: {metric}")
