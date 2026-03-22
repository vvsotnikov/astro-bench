#!/usr/bin/env python3
"""v81 evaluation: v36 arch with 50 epochs"""
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test
from verify import evaluate

X_test, f_test, y_test = load_test()

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.muon_conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.muon_bn1 = nn.BatchNorm2d(32)
        self.muon_res1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32))
        self.muon_res2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.muon_skip2 = nn.Sequential(nn.Conv2d(32, 64, 1, 2), nn.BatchNorm2d(64))
        self.muon_pool = nn.AdaptiveAvgPool2d(1)
        self.electron_conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.electron_bn1 = nn.BatchNorm2d(16)
        self.electron_res1 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16))
        self.electron_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(nn.Linear(85, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, matrices, features):
        m = torch.relu(self.muon_bn1(self.muon_conv1(matrices[:, 1:2])))
        m = m + self.muon_res1(m)
        m = torch.relu(self.muon_skip2(m) + self.muon_res2(m))
        m = self.muon_pool(m).view(m.size(0), -1)
        e = torch.relu(self.electron_bn1(self.electron_conv1(matrices[:, 0:1])))
        e = e + self.electron_res1(e)
        e = self.electron_pool(e).view(e.size(0), -1)
        return self.fusion(torch.cat([m, e, features], dim=1))

model = ResNet().cuda()
model.load_state_dict(torch.load('model_v81.pt'))
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
np.savez_compressed('predictions_v81.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v81: v36 arch with 50 epochs")
print(f"Metric: {metric}")
