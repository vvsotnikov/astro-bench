#!/usr/bin/env python3
"""Quick training template: fewer epochs to complete faster"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

def main(model_name, seed, fusion_dims=[256, 128]):
    """Quick training template"""
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
    perm = np.random.RandomState(seed).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = GammaDataset(matrices[train_idx], features[train_idx], labels[train_idx])
    val_ds = GammaDataset(matrices[val_idx], features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.muon_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.muon_bn1 = nn.BatchNorm2d(32)
            self.muon_res1 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32)
            )
            self.muon_res2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
            )
            self.muon_skip2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, stride=2), nn.BatchNorm2d(64))
            self.muon_pool = nn.AdaptiveAvgPool2d(1)
            self.electron_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.electron_bn1 = nn.BatchNorm2d(16)
            self.electron_res1 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16)
            )
            self.electron_pool = nn.AdaptiveAvgPool2d(1)
            # Configurable fusion network
            layers = []
            input_dim = 64 + 16 + 5
            for i, dim in enumerate(fusion_dims):
                layers.append(nn.Linear(input_dim, dim))
                layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
                input_dim = dim
            layers.append(nn.Linear(input_dim, 2))
            self.fusion = nn.Sequential(*layers)

        def forward(self, matrices, features):
            m = torch.relu(self.muon_bn1(self.muon_conv1(matrices[:, 1:2])))
            m = m + self.muon_res1(m)
            m = torch.relu(self.muon_skip2(m) + self.muon_res2(m))
            m = self.muon_pool(m).view(m.size(0), -1)
            e = torch.relu(self.electron_bn1(self.electron_conv1(matrices[:, 0:1])))
            e = e + self.electron_res1(e)
            e = self.electron_pool(e).view(e.size(0), -1)
            x = torch.cat([m, e, features], dim=1)
            return self.fusion(x)

    model = ResNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 20.0]).cuda())

    best_val_loss = float('inf')
    for epoch in range(20):
        model.train()
        train_loss = sum(
            (model.train() or True,
             criterion(model(m.cuda(), f.cuda()), l.cuda()).item()
             for m, f, l in train_loader)
            for _ in [None]
        )
        # Simpler version
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
                val_loss += criterion(model(mat.cuda(), feat.cuda()), label.cuda()).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'model_{model_name}.pt')
        scheduler.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{20}: train={train_loss:.4f} val={val_loss:.4f}")

    model.load_state_dict(torch.load(f'model_{model_name}.pt'))
    model.eval()

    # Batch inference
    all_scores = []
    batch_size = 512
    X_test_np = X_test.astype(np.float32)
    f_test_np = f_test.astype(np.float32)

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            end_i = min(i + batch_size, len(X_test))
            X_b = torch.from_numpy(X_test_np[i:end_i]).permute(0, 3, 1, 2).cuda()
            f_b = torch.from_numpy(f_test_np[i:end_i]).cuda()
            scores = torch.softmax(model(X_b, f_b), dim=1)[:, 0].cpu().numpy()
            all_scores.append(scores)

    gamma_scores = np.concatenate(all_scores)
    np.savez_compressed(f'predictions_{model_name}.npz', gamma_scores=gamma_scores)
    metric = evaluate(gamma_scores, f"{model_name}: Quick v36 variant")
    print(f"Metric: {metric}")

if __name__ == '__main__':
    # Determine which variant to run based on script name
    import sys
    if len(sys.argv) > 1:
        variant = sys.argv[1]
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        main(f'v{variant}', seed)
    else:
        print("Usage: python quick_train.py VARIANT_NUM [SEED]")
