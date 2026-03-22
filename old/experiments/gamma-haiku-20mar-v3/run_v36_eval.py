#!/usr/bin/env python3
"""Run evaluation on v36 model to test inference"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test
from verify import evaluate

print("Loading test data...", flush=True)
X_test, f_test, y_test = load_test()
print(f"Test data shapes: {X_test.shape}, {f_test.shape}", flush=True)

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

print("Creating model...", flush=True)
model = ResNet().cuda()

print("Loading v36 model weights...", flush=True)
model.load_state_dict(torch.load('model_v36.pt'))
print("Model loaded.", flush=True)

model.eval()
print("Running inference on test set...", flush=True)

# Process in batches
batch_size = 512
all_scores = []

X_test_np = X_test.astype(np.float32)
f_test_np = f_test.astype(np.float32)

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        end_i = min(i + batch_size, len(X_test))
        batch_x = X_test_np[i:end_i]
        batch_f = f_test_np[i:end_i]

        X_batch = torch.from_numpy(batch_x).permute(0, 3, 1, 2).cuda()
        f_batch = torch.from_numpy(batch_f).cuda()

        logits = model(X_batch, f_batch)
        scores = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
        all_scores.append(scores)

        print(f"  Processed {end_i}/{len(X_test)} samples", flush=True)

gamma_scores = np.concatenate(all_scores)
print(f"Total predictions: {len(gamma_scores)}", flush=True)

print("Saving predictions...", flush=True)
np.savez_compressed('predictions_v36_eval.npz', gamma_scores=gamma_scores)

print("Evaluating...", flush=True)
metric = evaluate(gamma_scores, "v36_eval: re-evaluate v36 with batch inference")
print(f"Metric: {metric}", flush=True)
