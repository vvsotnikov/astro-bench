"""v2: Improved CNN+Attn+MLP with:
- Fine-tuning on quality-cut subset after phase 1
- Data augmentation (random 90-degree rotations of 16x16 grids)
- More feature engineering
- Train on all 5.5M, then fine-tune on quality-cut events
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import time
import gc

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-composition-mar14"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 10  # fine-tune on quality-cut events
LR = 1e-3
LR_FINETUNE = 2e-4
LABEL_SMOOTH = 0.05
SEED = 42

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    """Extended feature engineering with 18 features."""
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    feats = [
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)),
        np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)),
        np.cos(np.radians(Az)),
        Ne - Nmu,                    # log(Ne/Nmu) — strongest discriminant
        Ne + Nmu,                    # total shower size
        (Ne - Nmu) / (Ne + Nmu + 1e-6),  # normalized ratio
        Ne - E,                      # electron excess relative to energy
        Nmu - E,                     # muon excess relative to energy
        # New features
        Ne * np.cos(np.radians(Ze)),  # zenith-corrected Ne
        Nmu * np.cos(np.radians(Ze)), # zenith-corrected Nmu
        (Ne - Nmu) * np.cos(np.radians(Ze)),  # zenith-corrected ratio
        E - 0.9 * Ne - 0.1 * Nmu,   # energy reconstruction residual (approx)
        np.sin(np.radians(2 * Az)),  # geomagnetic encoding
    ]
    return np.stack(feats, axis=1).astype(np.float32)


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
    def __init__(self, n_feat=18, n_classes=5):
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


def augment_batch(mat):
    """Random 90-degree rotations of detector grid."""
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        mat = torch.rot90(mat, k, [2, 3])
    return mat


def load_all_f32(split, feat_stats=None):
    p(f"Loading {split} data as float32 tensors...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    chunk = 250000
    mat_list = []
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        m = np.array(matrices[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
        if (i // chunk) % 4 == 0:
            p(f"  {split}: {end}/{n}")
    mat_tensor = torch.cat(mat_list, dim=0)
    del mat_list; gc.collect()

    feat_chunks = []
    for i in range(0, n, 500000):
        end = min(i + 500000, n)
        f = np.array(raw_feats[i:end], dtype=np.float32)
        feat_chunks.append(engineer_features(f))
    feats = np.concatenate(feat_chunks)
    del feat_chunks; gc.collect()

    if feat_stats is None:
        feat_mean = feats.mean(0)
        feat_std = feats.std(0) + 1e-6
    else:
        feat_mean, feat_std = feat_stats
    feats = (feats - feat_mean) / feat_std
    feat_tensor = torch.from_numpy(feats)
    del feats; gc.collect()

    label_tensor = torch.from_numpy(np.array(labels[:], dtype=np.int64))

    p(f"  {split}: mat={mat_tensor.shape} ({mat_tensor.element_size()*mat_tensor.nelement()/1e9:.1f}GB)")
    return mat_tensor, feat_tensor, label_tensor, (feat_mean, feat_std)


def get_quality_cut_mask(raw_feats_path, n):
    """Identify events passing quality cuts: Ze<30, Ne>4.8"""
    raw_feats = np.load(raw_feats_path, mmap_mode='r')
    # Load in chunks to avoid RAM issues
    chunk = 500000
    mask = np.zeros(n, dtype=bool)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        f = np.array(raw_feats[i:end], dtype=np.float32)
        Ze = f[:, 1]
        Ne = f[:, 3]
        mask[i:end] = (Ze < 30) & (Ne > 4.8)
    return mask


def train_epoch(model, loader, optimizer, criterion, scaler, augment=False):
    model.train()
    correct, total = 0, 0
    for mat_b, feat_b, label_b in loader:
        mat_b = mat_b.to(DEVICE)
        feat_b = feat_b.to(DEVICE)
        label_b = label_b.to(DEVICE)
        if augment:
            mat_b = augment_batch(mat_b)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            out = model(mat_b, feat_b)
            loss = criterion(out, label_b)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        correct += (out.argmax(1) == label_b).sum().item()
        total += len(label_b)
    return correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for mat_b, feat_b, label_b in loader:
            mat_b = mat_b.to(DEVICE)
            feat_b = feat_b.to(DEVICE)
            label_b = label_b.to(DEVICE)
            with autocast(device_type='cuda'):
                out = model(mat_b, feat_b)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
            correct += (out.argmax(1) == label_b).sum().item()
            total += len(label_b)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    acc = correct / total
    return preds, probs, acc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    # Load all data
    mat_train, feat_train, y_train, stats = load_all_f32("train")
    mat_test, feat_test, y_test, _ = load_all_f32("test", feat_stats=stats)
    n_train = len(y_train)

    # Create loaders
    train_loader = DataLoader(
        TensorDataset(mat_train, feat_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(mat_test, feat_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridModel(n_feat=feat_train.shape[1]).to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    # === Phase 1: Train on all data with augmentation ===
    p(f"\n{'='*60}")
    p(f"Phase 1: Full data training ({EPOCHS_PHASE1} epochs)")
    p(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1)

    best_acc = 0
    best_preds = None
    best_probs = None

    for epoch in range(EPOCHS_PHASE1):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, augment=True)
        preds, probs, test_acc = evaluate(model, test_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        p(f"Ep {epoch+1}/{EPOCHS_PHASE1}: train={train_acc:.4f} test={test_acc:.4f} lr={lr:.6f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = preds
            best_probs = probs
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v2_phase1.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    p(f"Phase 1 best: {best_acc:.4f}")

    # Save phase 1 predictions
    np.savez(f"{OUT_DIR}/predictions_v2_phase1.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v2_phase1.npy", best_probs)

    # === Phase 2: Fine-tune on quality-cut events ===
    p(f"\n{'='*60}")
    p(f"Phase 2: Fine-tuning on quality-cut events ({EPOCHS_PHASE2} epochs)")
    p(f"{'='*60}")

    # Load best model from phase 1
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v2_phase1.pt", weights_only=True))

    # Get quality cut mask for training data
    qc_mask = get_quality_cut_mask(f"{DATA_DIR}/composition_train/features.npy", n_train)
    n_qc = qc_mask.sum()
    p(f"Quality-cut events: {n_qc} ({n_qc/n_train*100:.1f}%)")

    # Create quality-cut loader
    qc_indices = np.where(qc_mask)[0]
    mat_qc = mat_train[qc_indices]
    feat_qc = feat_train[qc_indices]
    y_qc = y_train[qc_indices]

    qc_loader = DataLoader(
        TensorDataset(mat_qc, feat_qc, y_qc),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Fine-tune with lower LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)

    for epoch in range(EPOCHS_PHASE2):
        train_acc = train_epoch(model, qc_loader, optimizer, criterion, scaler, augment=True)
        preds, probs, test_acc = evaluate(model, test_loader)
        scheduler.step()
        p(f"FT {epoch+1}/{EPOCHS_PHASE2}: train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = preds
            best_probs = probs
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v2_phase2.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    # Save final
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v2.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: CNN+Attn+MLP 18feats, augment, phase1(20ep)+phase2(10ep QC finetune)")


if __name__ == "__main__":
    main()
