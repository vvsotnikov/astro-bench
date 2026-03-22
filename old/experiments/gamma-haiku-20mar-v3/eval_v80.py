#!/usr/bin/env python3
"""v80 evaluation: Simple MLP inference"""
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test
from verify import evaluate

X_test, f_test, y_test = load_test()

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
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
        spatial = matrices.reshape(matrices.size(0), -1)
        spatial_feat = self.spatial_mlp(spatial)
        x = torch.cat([spatial_feat, features], dim=1)
        return self.fusion(x)

model = SimpleMLP().cuda()
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
