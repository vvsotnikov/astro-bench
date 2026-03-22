import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from load_data import load_train, load_test
from verify import evaluate

torch.manual_seed(789)
np.random.seed(789)

class AttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)
    def forward(self, x):
        attn = torch.sigmoid(self.attn(x))
        weighted = x * attn
        return weighted.sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)

class DualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.attn = AttentionPool(64)
        self.mlp = nn.Sequential(nn.Linear(64 + 4, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
    def forward(self, x, scalar):
        x = x.permute(0, 3, 1, 2)
        feat = self.conv(x)
        feat = self.attn(feat)
        x = torch.cat([feat, scalar], dim=1)
        return self.mlp(x)

X_train, f_train, y_train = load_train()
X_test, f_test, y_test = load_test()
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
x_mean, x_std = X_train.mean(), X_train.std()
X_train = (X_train - x_mean) / (x_std + 1e-6)
X_test = (X_test - x_mean) / (x_std + 1e-6)

scalar_train = np.stack([f_train[:, 3] - f_train[:, 4], f_train[:, 1] / 30, f_train[:, 0] / 18, np.log1p(f_train[:, 3])], axis=1).astype(np.float32)
scalar_test = np.stack([f_test[:, 3] - f_test[:, 4], f_test[:, 1] / 30, f_test[:, 0] / 18, np.log1p(f_test[:, 3])], axis=1).astype(np.float32)
s_mean, s_std = scalar_train.mean(axis=0), scalar_train.std(axis=0)
scalar_train = (scalar_train - s_mean) / (s_std + 1e-6)
scalar_test = (scalar_test - s_mean) / (s_std + 1e-6)

device = torch.device("cuda:0")
rng = np.random.default_rng(789)
n = len(y_train)
val_mask = rng.random(n) < 0.2
train_mask = ~val_mask

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train[train_mask]), torch.from_numpy(scalar_train[train_mask]), 
    torch.from_numpy(y_train[train_mask]).long()), batch_size=4096, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_train[val_mask]), torch.from_numpy(scalar_train[val_mask]), 
    torch.from_numpy(y_train[val_mask]).long()), batch_size=8192, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(scalar_test)), batch_size=8192)

model = DualCNN().to(device)
n_gamma = (y_train == 0).sum()
n_hadron = (y_train == 1).sum()
class_weights = torch.tensor([len(y_train) / (2 * n_gamma), len(y_train) / (2 * n_hadron)], dtype=torch.float32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)
criterion = nn.CrossEntropyLoss(weight=class_weights)

best_val_loss = float("inf")
for epoch in range(35):
    model.train()
    for x, s, y in train_loader:
        x, s, y = x.to(device), s.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(x, s), y).backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    val_loss = 0
    all_scores = []
    with torch.no_grad():
        for x, s, y in val_loader:
            x, s, y = x.to(device), s.to(device), y.to(device)
            logits = model(x, s)
            val_loss += criterion(logits, y).item() * len(y)
            all_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())
    val_loss /= len(y_train[val_mask])
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_scores = np.concatenate(all_scores)

model.eval()
test_scores = []
with torch.no_grad():
    for x, s in test_loader:
        x, s = x.to(device), s.to(device)
        test_scores.append(torch.softmax(model(x, s), 1)[:, 0].cpu().numpy())

np.savez("predictions_v17.npz", gamma_scores=np.concatenate(test_scores))
evaluate(np.concatenate(test_scores), "v17: v3 with seed=789 (ensemble seed #3)")
