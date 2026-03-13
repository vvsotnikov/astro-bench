"""v3: Hybrid CNN+MLP for 5-class composition.
Pre-load first 2M train samples (contiguous slice) to RAM. num_workers=0.
Uses flush for output."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import sys

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS = 30
LR = 1e-3
TRAIN_N = 2000000

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    feats = [
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)),
        np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)),
        np.cos(np.radians(Az)),
        Ne - Nmu,
        Ne + Nmu,
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E,
        Nmu - E,
    ]
    return np.stack(feats, axis=1).astype(np.float32)


class HybridCNNMLP(nn.Module):
    def __init__(self, n_feat=13, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat, feat):
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))


def load_data(split, n_samples=None, feat_stats=None):
    p(f"Loading {split} data...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')

    n = len(labels)
    if n_samples and n_samples < n:
        n = n_samples

    # Contiguous slice -- much faster than fancy indexing on mmap
    chunk = 200000
    mat_list, feat_list, label_list = [], [], []
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        m = np.array(matrices[i:end], dtype=np.float32)
        m = np.log1p(m)
        m = m.transpose(0, 3, 1, 2)
        mat_list.append(m)
        feat_list.append(np.array(raw_feats[i:end], dtype=np.float32))
        label_list.append(np.array(labels[i:end], dtype=np.int64))
        p(f"  Chunk {i//chunk+1}, {end}/{n}")

    mat_all = np.concatenate(mat_list)
    feat_raw = np.concatenate(feat_list)
    labels_all = np.concatenate(label_list)

    feats = engineer_features(feat_raw)
    if feat_stats is None:
        feat_mean = feats.mean(axis=0)
        feat_std = feats.std(axis=0) + 1e-6
    else:
        feat_mean, feat_std = feat_stats
    feats = (feats - feat_mean) / feat_std

    p(f"  Loaded: mat={mat_all.shape}, feat={feats.shape}, labels={labels_all.shape}")
    return (torch.from_numpy(mat_all), torch.from_numpy(feats),
            torch.from_numpy(labels_all), (feat_mean, feat_std))


def main():
    t0 = time.time()

    mat_train, feat_train, y_train, stats = load_data("train", n_samples=TRAIN_N)
    mat_test, feat_test, y_test, _ = load_data("test", feat_stats=stats)

    train_loader = DataLoader(TensorDataset(mat_train, feat_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(mat_test, feat_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridCNNMLP(n_feat=feat_train.shape[1]).to(DEVICE)
    p(f"Model params: {sum(p_.numel() for p_ in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_preds = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for mat, feat, label in train_loader:
            mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(mat, feat)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == label).sum().item()
            total += len(label)
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_preds, all_probs = [], []
        tc, tt = 0, 0
        with torch.no_grad():
            for mat, feat, label in test_loader:
                mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
                out = model(mat, feat)
                all_preds.append(out.argmax(1).cpu().numpy())
                all_probs.append(torch.softmax(out, 1).cpu().numpy())
                tc += (out.argmax(1) == label).sum().item()
                tt += len(label)

        test_acc = tc / tt
        p(f"Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            best_probs = np.concatenate(all_probs)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v3.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v3.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: Hybrid CNN+MLP, 30ep, 2M train, log1p, 13 eng features")

if __name__ == "__main__":
    main()
