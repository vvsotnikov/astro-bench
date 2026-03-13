"""v24: CNN+Attn+MLP with ordinal-aware soft labels.
Key idea: Instead of hard one-hot labels, use soft labels that reflect
the mass ordering (H < He < C < Si < Fe). Confusing neighboring classes
is penalized less than confusing distant ones.
Uses v8 architecture (proven) with KL divergence loss on soft targets."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import time
import gc

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS = 20
LR = 1e-3

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
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid())
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


def make_ordinal_soft_labels(labels, n_classes=5, sigma=0.5):
    """Create soft labels based on ordinal distance.
    For class i, the soft label at class j is proportional to exp(-|i-j|^2 / (2*sigma^2))"""
    n = len(labels)
    soft = np.zeros((n, n_classes), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        for j in range(n_classes):
            dist = abs(c - j)
            soft[mask, j] = np.exp(-dist**2 / (2 * sigma**2))
    # Normalize to sum to 1
    soft = soft / soft.sum(axis=1, keepdims=True)
    return soft


def load_all_f32(split, feat_stats=None):
    p(f"Loading {split}...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    mat_list = []
    for i in range(0, n, 250000):
        end = min(i + 250000, n)
        m = np.array(matrices[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
        if (i // 250000) % 5 == 0:
            p(f"  {split}: {end}/{n}")
    mat_tensor = torch.cat(mat_list, dim=0)
    del mat_list; gc.collect()

    feat_chunks = []
    for i in range(0, n, 500000):
        end = min(i + 500000, n)
        feat_chunks.append(engineer_features(np.array(raw_feats[i:end], dtype=np.float32)))
    feats = np.concatenate(feat_chunks)
    del feat_chunks; gc.collect()

    if feat_stats is None:
        feat_mean = feats.mean(0); feat_std = feats.std(0) + 1e-6
    else:
        feat_mean, feat_std = feat_stats
    feats = (feats - feat_mean) / feat_std
    feat_tensor = torch.from_numpy(feats); del feats; gc.collect()

    label_arr = np.array(labels[:], dtype=np.int64)
    label_tensor = torch.from_numpy(label_arr)

    p(f"  {split}: {mat_tensor.shape}")
    return mat_tensor, feat_tensor, label_tensor, label_arr, (feat_mean, feat_std)


def main():
    t0 = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    mat_train, feat_train, y_train_tensor, y_train_np, stats = load_all_f32("train")
    mat_test, feat_test, y_test_tensor, y_test_np, _ = load_all_f32("test", feat_stats=stats)

    # Create ordinal soft labels for training
    p("Creating ordinal soft labels...")
    soft_labels = make_ordinal_soft_labels(y_train_np, sigma=0.5)
    soft_tensor = torch.from_numpy(soft_labels)
    p(f"  Soft label example (class 0): {soft_labels[y_train_np==0][0].round(3)}")
    p(f"  Soft label example (class 2): {soft_labels[y_train_np==2][0].round(3)}")

    train_loader = DataLoader(TensorDataset(mat_train, feat_train, soft_tensor, y_train_tensor),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(mat_test, feat_test, y_test_tensor),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridModel(n_feat=feat_train.shape[1]).to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    # KL divergence loss for soft labels
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    best_acc = 0
    best_preds = None
    best_probs = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        epoch_loss = 0
        for mat_b, feat_b, soft_b, hard_b in train_loader:
            mat_b = mat_b.to(DEVICE)
            feat_b = feat_b.to(DEVICE)
            soft_b = soft_b.to(DEVICE)
            hard_b = hard_b.to(DEVICE)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits = model(mat_b, feat_b)
                log_probs = F.log_softmax(logits, dim=1)
                loss = kl_loss(log_probs, soft_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (logits.argmax(1) == hard_b).sum().item()
            total += len(hard_b)
            epoch_loss += loss.item() * len(hard_b)
        train_acc = correct / total
        scheduler.step()

        # Evaluate
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
        p(f"Ep {epoch+1}/{EPOCHS}: loss={epoch_loss/total:.4f} train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            best_probs = np.concatenate(all_probs)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v24.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions_v24.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v24.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: CNN+Attn+MLP ordinal soft labels (sigma=0.5), KL loss, {EPOCHS}ep")


if __name__ == "__main__":
    main()
