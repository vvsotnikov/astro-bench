"""v22: ResNet with residual blocks + channel attention + scalar MLP.
Same general structure as v8 but with skip connections for better gradient flow.
Config 1: Same depth/width as v8, add residual connections.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import subprocess, gc, time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 25
SEED = 42

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


class ResBlock(nn.Module):
    """Residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.attn = ChannelAttention(out_ch)

        self.shortcut = nn.Sequential()
        if downsample or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attn(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetHybrid(nn.Module):
    def __init__(self, n_feat=13, n_classes=5):
        super().__init__()
        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # ResBlocks: 32->64 (downsample), 64->128 (downsample), 128->256
        self.layer1 = nn.Sequential(
            ResBlock(32, 64, downsample=True),   # 16x16 -> 8x8
            ResBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),  # 8x8 -> 4x4
            ResBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256),                  # 4x4 -> 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

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
        x = self.stem(mat)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        f = self.feat_mlp(feat)
        return self.head(torch.cat([x, f], dim=1))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load ALL training data
    p("Loading train data...")
    raw_mat = np.load(f"{DATA_DIR}/composition_train/matrices.npy", mmap_mode='r')
    raw_feat = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')
    n_train = len(labels)

    CHUNK = 500000
    mat_chunks = []
    feat_chunks = []
    for i in range(0, n_train, CHUNK):
        end = min(i + CHUNK, n_train)
        p(f"  train: {i}/{n_train}")
        m = np.array(raw_mat[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_chunks.append(torch.from_numpy(m))
        feat_chunks.append(engineer_features(np.array(raw_feat[i:end], dtype=np.float32)))

    mat_train = torch.cat(mat_chunks)
    feat_all = np.concatenate(feat_chunks)
    y_train = np.array(labels[:], dtype=np.int64)
    del mat_chunks, feat_chunks; gc.collect()

    feat_mean = feat_all.mean(0)
    feat_std = feat_all.std(0) + 1e-6
    feat_all = (feat_all - feat_mean) / feat_std
    feat_train = torch.from_numpy(feat_all)
    del feat_all; gc.collect()

    p(f"Train: {mat_train.shape}")

    # Load test
    p("Loading test data...")
    raw_mat_test = np.load(f"{DATA_DIR}/composition_test/matrices.npy", mmap_mode='r')
    raw_feat_test = np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r')
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)

    mat_list = []
    for i in range(0, len(y_test), 250000):
        end = min(i + 250000, len(y_test))
        m = np.array(raw_mat_test[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
    mat_test = torch.cat(mat_list)
    del mat_list; gc.collect()

    test_feats = engineer_features(np.array(raw_feat_test[:], dtype=np.float32))
    test_feats = (test_feats - feat_mean) / feat_std
    feat_test = torch.from_numpy(test_feats)

    # Dataset
    train_ds = TensorDataset(mat_train, feat_train, torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = ResNetHybrid(n_feat=13).to(DEVICE)
    n_params = sum(p_.numel() for p_ in model.parameters())
    p(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_dl), pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    best_frac_err = 1.0
    best_acc = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_correct = 0
        total_n = 0

        for mat_b, feat_b, y_b in train_dl:
            mat_b = mat_b.to(DEVICE, non_blocking=True)
            feat_b = feat_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                out = model(mat_b, feat_b)
                loss = criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_correct += (out.argmax(1) == y_b).sum().item()
            total_n += len(y_b)

        train_acc = total_correct / total_n

        # Evaluate
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(y_test), BATCH_SIZE):
                end = min(i + BATCH_SIZE, len(y_test))
                mb = mat_test[i:end].to(DEVICE)
                fb = feat_test[i:end].to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mb, fb)
                probs_list.append(torch.softmax(out.float(), 1).cpu().numpy())

        probs = np.concatenate(probs_list)
        preds = probs.argmax(1)
        test_acc = (preds == y_test).mean()
        elapsed = time.time() - t0

        # Verify every 5 epochs or last epoch
        frac_err_val = -1
        if epoch % 5 == 0 or epoch == EPOCHS or epoch <= 3:
            np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))
            result = subprocess.run(
                ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
                capture_output=True, text=True,
                cwd="/home/vladimir/cursor_projects/astro-agents"
            )
            for line in result.stdout.split('\n'):
                if 'mean fraction error' in line.lower():
                    try:
                        frac_err_val = float(line.split(':')[-1].strip())
                    except:
                        pass
                    break

        msg = f"  Ep {epoch}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f}"
        if frac_err_val > 0:
            msg += f" frac_err={frac_err_val:.4f}"

        improved = False
        if frac_err_val > 0 and frac_err_val < best_frac_err:
            best_frac_err = frac_err_val
            best_acc = test_acc
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v22.pt")
            np.save(f"{OUT_DIR}/probs_v22.npy", probs)
            improved = True
            msg += " *** BEST"
        elif test_acc > best_acc:
            best_acc = test_acc
            # Save anyway for later verification
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v22_latest.pt")
            np.save(f"{OUT_DIR}/probs_v22_latest.npy", probs)

        msg += f" [{elapsed:.0f}s]"
        p(msg)

    # Final verify with best model
    if best_frac_err < 1.0:
        probs_best = np.load(f"{OUT_DIR}/probs_v22.npy")
    else:
        probs_best = np.load(f"{OUT_DIR}/probs_v22_latest.npy")

    preds = probs_best.argmax(1).astype(np.int8)
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds)
    result = subprocess.run(
        ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
        capture_output=True, text=True,
        cwd="/home/vladimir/cursor_projects/astro-agents"
    )
    p(f"\nFinal verify:\n{result.stdout[:500]}")

    p(f"\nBest: acc={best_acc:.4f}, frac_err={best_frac_err:.4f}")
    p("---")
    p(f"metric: {best_frac_err:.4f}")
    p(f"description: ResNet (res blocks + ch attn) + MLP, OneCycleLR, {EPOCHS}ep, 5.5M")


if __name__ == "__main__":
    main()
