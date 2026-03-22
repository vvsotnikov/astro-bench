import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from load_data import load_train, load_test
from verify import evaluate

class AttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)
    def forward(self, x):
        attn = torch.sigmoid(self.attn(x))
        return (x * attn).sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)

class DualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.attn = AttentionPool(64)
        self.mlp = nn.Sequential(nn.Linear(68, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
    def forward(self, x, s):
        x = self.attn(self.conv(x.permute(0, 3, 1, 2)))
        return self.mlp(torch.cat([x, s], 1))

X_train, f_train, y_train = load_train()
X_test, f_test, y_test = load_test()
X_train = ((X_train.astype(np.float32) - X_train.mean()) / (X_train.std() + 1e-6))
X_test = ((X_test.astype(np.float32) - X_test.mean()) / (X_test.std() + 1e-6))
s_train = np.stack([f_train[:, 3] - f_train[:, 4], f_train[:, 1] / 30, f_train[:, 0] / 18, np.log1p(f_train[:, 3])], 1).astype(np.float32)
s_test = np.stack([f_test[:, 3] - f_test[:, 4], f_test[:, 1] / 30, f_test[:, 0] / 18, np.log1p(f_test[:, 3])], 1).astype(np.float32)
s_train = ((s_train - s_train.mean(0)) / (s_train.std(0) + 1e-6))
s_test = ((s_test - s_test.mean(0)) / (s_test.std(0) + 1e-6))

dev = torch.device('cuda:0')
rng = np.random.default_rng(42)
vm = rng.random(len(y_train)) < 0.2
tl = DataLoader(TensorDataset(torch.from_numpy(X_train[~vm]), torch.from_numpy(s_train[~vm]), torch.from_numpy(y_train[~vm]).long()), 4096, True)
vl = DataLoader(TensorDataset(torch.from_numpy(X_train[vm]), torch.from_numpy(s_train[vm]), torch.from_numpy(y_train[vm]).long()), 8192, False)
tel = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(s_test)), 8192)

m = DualCNN().to(dev)
ng, nh = (y_train == 0).sum(), (y_train == 1).sum()
cw = torch.tensor([len(y_train) / (2 * ng), len(y_train) / (2 * nh)], dtype=torch.float32).to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)  # 50 epochs instead of 35
crit = nn.CrossEntropyLoss(weight=cw)

bvl = float('inf')
for _ in range(50):  # 50 epochs
    m.train()
    for x, s, y in tl:
        opt.zero_grad()
        crit(m(x.to(dev), s.to(dev)), y.to(dev)).backward()
        opt.step()
    sch.step()
    m.eval()
    vl_sum = 0
    with torch.no_grad():
        for x, s, y in vl:
            x, s, y = x.to(dev), s.to(dev), y.to(dev)
            vl_sum += crit(m(x, s), y).item() * len(y)
    if vl_sum / len(y_train[vm]) < bvl:
        bvl = vl_sum / len(y_train[vm])

m.eval()
ts = []
with torch.no_grad():
    for x, s in tel:
        x, s = x.to(dev), s.to(dev)
        ts.append(torch.softmax(m(x, s), 1)[:, 0].cpu().numpy())

np.savez('predictions_v24.npz', gamma_scores=np.concatenate(ts))
evaluate(np.concatenate(ts), 'v24: v3 with 50 epochs (longer training)')
