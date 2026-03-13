"""v5: Hybrid CNN+Attn+MLP with AMP for speed.
2M train (1M each sim), mixed precision, 40 epochs."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS = 30
LR = 1e-3

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


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.ReLU(),
            nn.Linear(ch // r, ch), nn.Sigmoid(),
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


def load_contiguous(matrices, feats, labels, start, end):
    m = np.array(matrices[start:end], dtype=np.float32)
    m = np.log1p(m).transpose(0, 3, 1, 2)
    f = np.array(feats[start:end], dtype=np.float32)
    l = np.array(labels[start:end], dtype=np.int64)
    return m, f, l


def main():
    t0 = time.time()

    p("Loading training data...")
    matrices = np.load(f"{DATA_DIR}/composition_train/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')
    n_total = len(labels)

    # 1M QGSJet + 1M EPOS = 2M total
    n_each = 1000000
    chunk = 250000
    mat_list, feat_list, label_list = [], [], []

    for i in range(0, n_each, chunk):
        m, f, l = load_contiguous(matrices, raw_feats, labels, i, min(i+chunk, n_each))
        mat_list.append(m); feat_list.append(f); label_list.append(l)
        p(f"  QGS: {min(i+chunk, n_each)}/{n_each}")

    epos_start = n_total - n_each
    for i in range(epos_start, n_total, chunk):
        end = min(i+chunk, n_total)
        m, f, l = load_contiguous(matrices, raw_feats, labels, i, end)
        mat_list.append(m); feat_list.append(f); label_list.append(l)
        p(f"  EPOS: {end-epos_start}/{n_each}")

    mat_all = np.concatenate(mat_list)
    feat_raw = np.concatenate(feat_list)
    labels_all = np.concatenate(label_list)
    p(f"  Total: {len(labels_all)}")

    feats = engineer_features(feat_raw)
    feat_mean = feats.mean(0); feat_std = feats.std(0) + 1e-6
    feats = (feats - feat_mean) / feat_std

    # Test
    p("Loading test data...")
    t_mat = np.load(f"{DATA_DIR}/composition_test/matrices.npy", mmap_mode='r')
    t_feat = np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r')
    t_lab = np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r')
    n_test = len(t_lab)
    m_test = np.log1p(np.array(t_mat[:], dtype=np.float32)).transpose(0, 3, 1, 2)
    f_test = (engineer_features(np.array(t_feat[:], dtype=np.float32)) - feat_mean) / feat_std
    l_test = np.array(t_lab[:], dtype=np.int64)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(mat_all), torch.from_numpy(feats), torch.from_numpy(labels_all)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(m_test), torch.from_numpy(f_test), torch.from_numpy(l_test)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridModel(n_feat=feats.shape[1]).to(DEVICE)
    p(f"Params: {sum(pp.numel() for pp in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0
    best_preds = None
    best_probs = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for mat_b, feat_b, label_b in train_loader:
            mat_b, feat_b, label_b = mat_b.to(DEVICE), feat_b.to(DEVICE), label_b.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                out = model(mat_b, feat_b)
                loss = criterion(out, label_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (out.argmax(1) == label_b).sum().item()
            total += len(label_b)
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_preds, all_probs = [], []
        tc, tt = 0, 0
        with torch.no_grad():
            for mat_b, feat_b, label_b in test_loader:
                mat_b, feat_b, label_b = mat_b.to(DEVICE), feat_b.to(DEVICE), label_b.to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mat_b, feat_b)
                all_preds.append(out.argmax(1).cpu().numpy())
                all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
                tc += (out.argmax(1) == label_b).sum().item()
                tt += len(label_b)

        test_acc = tc / tt
        p(f"Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            best_probs = np.concatenate(all_probs)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v5.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v5.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: CNN+Attn+MLP AMP, 40ep, 2M train (both sims), 13 eng feats")

if __name__ == "__main__":
    main()
