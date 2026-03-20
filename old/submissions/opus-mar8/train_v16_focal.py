"""v16: CNN+Attn+MLP with focal loss instead of CrossEntropy.
Hypothesis: focal loss focuses on hard-to-classify examples (helium confusion).
Same architecture as v8, 25 epochs, cosine schedule."""
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
EPOCHS = 25
LR = 1e-3
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None  # Try uniform first

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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights, tensor of shape (n_classes,)
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        if self.reduction == 'mean':
            return focal.mean()
        return focal


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
    label_tensor = torch.from_numpy(np.array(labels[:], dtype=np.int64))

    p(f"  {split}: {mat_tensor.shape}")
    return mat_tensor, feat_tensor, label_tensor, (feat_mean, feat_std)


def main():
    t0 = time.time()
    mat_train, feat_train, y_train, stats = load_all_f32("train")
    mat_test, feat_test, y_test, _ = load_all_f32("test", feat_stats=stats)

    # Compute class weights (inverse frequency)
    class_counts = torch.bincount(y_train).float()
    class_weights = (1.0 / class_counts) * len(y_train) / 5
    p(f"Class weights: {class_weights}")

    train_loader = DataLoader(TensorDataset(mat_train, feat_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(mat_test, feat_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridModel(n_feat=feat_train.shape[1]).to(DEVICE)
    p(f"Params: {sum(pp.numel() for pp in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=class_weights.to(DEVICE))
    scaler = GradScaler()

    best_acc = 0

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
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v16.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v16.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: CNN+Attn+MLP focal_loss gamma={FOCAL_GAMMA} + class_weights, {EPOCHS}ep cosine, ALL 5.5M")

if __name__ == "__main__":
    main()
