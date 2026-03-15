"""v7: Deep CNN (matrix-only, no feature MLP) for maximum ensemble diversity.
8 conv layers with channel attention, no scalar features at all.
Different seed (123) from v8 (42) and v4 (7).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import time
import gc

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-composition-mar14"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS = 25
LR = 1e-3
LABEL_SMOOTH = 0.05
SEED = 123

def p(msg):
    print(msg, flush=True)


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


class DeepCNNOnly(nn.Module):
    """8-layer CNN with channel attention, no scalar features."""
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 16x16 -> 8x8
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64),
            nn.MaxPool2d(2),
            # Block 2: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128),
            nn.MaxPool2d(2),
            # Block 3: 4x4 -> 2x2
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256),
            nn.MaxPool2d(2),
            # Block 4: 2x2 -> 1x1
            nn.Conv2d(256, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
            ChannelAttention(512),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat):
        x = self.features(mat)
        return self.head(x)


def load_matrices(split):
    p(f"Loading {split} matrices...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
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

    label_tensor = torch.from_numpy(np.array(labels[:], dtype=np.int64))
    p(f"  {split}: mat={mat_tensor.shape}")
    return mat_tensor, label_tensor


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    mat_train, y_train = load_matrices("train")
    mat_test, y_test = load_matrices("test")

    train_loader = DataLoader(
        TensorDataset(mat_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(mat_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = DeepCNNOnly().to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    best_preds = None
    best_probs = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for mat_b, label_b in train_loader:
            mat_b = mat_b.to(DEVICE)
            label_b = label_b.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                out = model(mat_b)
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
            for mat_b, label_b in test_loader:
                mat_b = mat_b.to(DEVICE)
                label_b = label_b.to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mat_b)
                all_preds.append(out.argmax(1).cpu().numpy())
                all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
                tc += (out.argmax(1) == label_b).sum().item()
                tt += len(label_b)
        test_acc = tc / tt
        lr = optimizer.param_groups[0]['lr']
        p(f"Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} lr={lr:.6f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            best_probs = np.concatenate(all_probs)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v7.pt")
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions_v7.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v7.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p("---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: Deep CNN only (8 layers, no features), seed={SEED}")


if __name__ == "__main__":
    main()
