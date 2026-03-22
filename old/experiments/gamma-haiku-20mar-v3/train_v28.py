#!/usr/bin/env python3
"""v28: Ensemble of top 3 single models (v2, v4, v1)"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_test
from verify import evaluate

# Load test data
X_test, f_test, y_test = load_test()

# Models: v2 (ResNet), v4 (MuonCNN+Focal), v1 (MuonCNN)
class MuonCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.muon_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.muon_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.muon_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.electron_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.electron_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fusion = nn.Sequential(
            nn.Linear(64*4*4 + 16*4*4 + 5, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, matrices, features):
        muon = matrices[:, 1:2]
        electron = matrices[:, 0:1]
        m = torch.relu(self.muon_conv1(muon))
        m = torch.relu(self.muon_conv2(m))
        m = self.muon_pool(m).view(m.size(0), -1)
        e = torch.relu(self.electron_conv1(electron))
        e = self.electron_pool(e).view(e.size(0), -1)
        x = torch.cat([m, e, features], dim=1)
        return self.fusion(x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.muon_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.muon_res = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.muon_res2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.muon_skip2 = nn.Conv2d(32, 64, kernel_size=1, stride=2)
        self.muon_pool = nn.AdaptiveAvgPool2d(1)
        self.electron_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.electron_res = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1)
        )
        self.electron_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 16 + 5, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, matrices, features):
        muon = matrices[:, 1:2]
        electron = matrices[:, 0:1]
        m = torch.relu(self.muon_conv1(muon))
        m = m + self.muon_res(m)
        m = torch.relu(self.muon_skip2(m) + self.muon_res2(m))
        m = self.muon_pool(m).view(m.size(0), -1)
        e = torch.relu(self.electron_conv1(electron))
        e = e + self.electron_res(e)
        e = self.electron_pool(e).view(e.size(0), -1)
        x = torch.cat([m, e, features], dim=1)
        return self.fusion(x)

# Load trained models
X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).permute(0, 3, 1, 2)
f_test_tensor = torch.from_numpy(f_test.astype(np.float32))

models_dict = {
    'v2': ResNet(),
    'v4': MuonCNN(),
    'v1': MuonCNN()
}

try:
    models_dict['v2'].load_state_dict(torch.load('model_v2.pt'))
    models_dict['v4'].load_state_dict(torch.load('model_v4.pt'))
    models_dict['v1'].load_state_dict(torch.load('model_v1.pt'))
    print("✓ Loaded all 3 models")
except Exception as e:
    print(f"Error loading models: {e}")
    import sys
    sys.exit(1)

for model in models_dict.values():
    model.eval()

# Get predictions from all 3
all_scores = []
with torch.no_grad():
    for name, model in models_dict.items():
        logits = model(X_test_tensor, f_test_tensor)
        scores = torch.softmax(logits, dim=1)[:, 0].numpy()
        all_scores.append(scores)
        print(f"{name}: mean={scores.mean():.4f}")

# Ensemble: equal weights
ensemble_scores = np.mean(all_scores, axis=0)
np.savez_compressed('predictions_v28.npz', gamma_scores=ensemble_scores)
metric = evaluate(ensemble_scores, "v28: Equal-weight ensemble (v2 + v4 + v1)")
print(f"Ensemble metric: {metric}")
