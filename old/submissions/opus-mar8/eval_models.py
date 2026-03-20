"""Evaluate all saved model checkpoints on test set."""
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import subprocess
import gc
import os

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    return np.stack([
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu, (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
    ], axis=1).astype(np.float32)

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class HybridModel(nn.Module):
    def __init__(self, n_feat=13, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )
    def forward(self, mat, feat):
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))

# Load test data
p("Loading test data...")
matrices = np.load(f"{DATA_DIR}/composition_test/matrices.npy", mmap_mode='r')
raw_feats = np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r')
labels = np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r')
n = len(labels)

mat_list = []
for i in range(0, n, 250000):
    end = min(i + 250000, n)
    m = np.array(matrices[i:end], dtype=np.float32)
    m = np.log1p(m).transpose(0, 3, 1, 2)
    mat_list.append(torch.from_numpy(m))
mat_test = torch.cat(mat_list, dim=0)
del mat_list; gc.collect()

# Compute train stats for normalization
raw_train = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
n_train = len(raw_train)
chunks = []
for i in range(0, n_train, 500000):
    end = min(i + 500000, n_train)
    chunks.append(engineer_features(np.array(raw_train[i:end], dtype=np.float32)))
train_feats = np.concatenate(chunks)
feat_mean = train_feats.mean(0)
feat_std = train_feats.std(0) + 1e-6
del train_feats, chunks; gc.collect()

test_feats = engineer_features(np.array(raw_feats[:], dtype=np.float32))
test_feats = (test_feats - feat_mean) / feat_std
feat_test = torch.from_numpy(test_feats)
y_test = np.array(labels[:], dtype=np.int64)

p(f"Test data: {mat_test.shape}, features: {feat_test.shape}")

# Models that use the standard HybridModel architecture with 13 features
standard_models = ["model_v8.pt", "model_v9.pt", "model_v11.pt", "model_v16.pt"]

for model_file in standard_models:
    path = f"{OUT_DIR}/{model_file}"
    if not os.path.exists(path):
        p(f"{model_file}: NOT FOUND")
        continue

    model = HybridModel(n_feat=13).to(DEVICE)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    except Exception as e:
        p(f"{model_file}: LOAD ERROR: {e}")
        continue
    model.eval()

    probs_list = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n)
            mat_b = mat_test[i:end].to(DEVICE)
            feat_b = feat_test[i:end].to(DEVICE)
            with autocast(device_type='cuda'):
                out = model(mat_b, feat_b)
            probs_list.append(torch.softmax(out.float(), 1).cpu().numpy())

    probs = np.concatenate(probs_list)
    preds = probs.argmax(1)
    acc = (preds == y_test).mean()

    # Save, verify, restore
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))
    result = subprocess.run(
        ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
        capture_output=True, text=True, cwd="/home/vladimir/cursor_projects/astro-agents"
    )
    frac_err = "NOT FOUND"
    for line in result.stdout.split('\n'):
        if 'mean fraction error' in line.lower():
            frac_err = line.strip()
            break

    # Also save probs
    name = model_file.replace("model_", "").replace(".pt", "")
    np.save(f"{OUT_DIR}/probs_{name}_eval.npy", probs)

    p(f"{model_file:20s}: acc={acc:.4f}  {frac_err}")
    del model; torch.cuda.empty_cache(); gc.collect()

# Restore v8
probs_v8 = np.load(f"{OUT_DIR}/probs_v8.npy")
np.savez(f"{OUT_DIR}/predictions.npz", predictions=probs_v8.argmax(1).astype(np.int8))
p("\nRestored v8 predictions")
