"""Evaluate v37 base model and extract predictions."""
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import gc

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096

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
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class CNNHybrid(nn.Module):
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


def main():
    # Load test data
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
    mat_test = torch.cat(mat_list); del mat_list; gc.collect()

    # Need train stats for normalization
    raw_train_feats = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    n_train = len(raw_train_feats)
    feat_chunks = []
    for i in range(0, n_train, 500000):
        end = min(i + 500000, n_train)
        feat_chunks.append(engineer_features(np.array(raw_train_feats[i:end], dtype=np.float32)))
    train_feats = np.concatenate(feat_chunks)
    feat_mean = train_feats.mean(0); feat_std = train_feats.std(0) + 1e-6
    del train_feats, feat_chunks; gc.collect()

    test_feats = engineer_features(np.array(raw_feats[:], dtype=np.float32))
    test_feats = (test_feats - feat_mean) / feat_std
    feat_test = torch.from_numpy(test_feats)
    y_test = np.array(labels[:], dtype=np.int64)

    # Load model
    model = CNNHybrid(n_feat=13).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v37_base.pt", weights_only=True))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n)
            mb = mat_test[i:end].to(DEVICE)
            fb = feat_test[i:end].to(DEVICE)
            with autocast(device_type='cuda'):
                out = model(mb, fb)
            all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())

    probs = np.concatenate(all_probs)
    preds = probs.argmax(1)
    acc = (preds == y_test).mean()

    np.save(f"{OUT_DIR}/probs_v37.npy", probs)
    np.savez(f"{OUT_DIR}/predictions_v37.npz", predictions=preds.astype(np.int8))

    print(f"v37 base: acc={acc:.4f}", flush=True)
    print(f"---", flush=True)
    print(f"metric: {acc:.4f}", flush=True)
    print(f"description: v37 base model eval", flush=True)


if __name__ == "__main__":
    main()
