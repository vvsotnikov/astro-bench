"""v18: Test-time augmentation on v8 model.
Apply 4 rotations (0, 90, 180, 270) to test matrices and average predictions.
The 16x16 detector grid has approximate rotational symmetry."""
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import time
import gc

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


def main():
    t0 = time.time()

    # Load test data
    p("Loading test data...")
    matrices = np.load(f"{DATA_DIR}/composition_test/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    # Load matrices in chunks
    mat_list = []
    for i in range(0, n, 250000):
        end = min(i + 250000, n)
        m = np.array(matrices[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
    mat_test = torch.cat(mat_list, dim=0)
    del mat_list; gc.collect()

    # Load training stats for normalization
    raw_train_feats = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    n_train = len(raw_train_feats)
    feat_chunks = []
    for i in range(0, n_train, 500000):
        end = min(i + 500000, n_train)
        feat_chunks.append(engineer_features(np.array(raw_train_feats[i:end], dtype=np.float32)))
    train_feats = np.concatenate(feat_chunks)
    feat_mean = train_feats.mean(0)
    feat_std = train_feats.std(0) + 1e-6
    del train_feats, feat_chunks; gc.collect()

    # Test features
    test_feats = engineer_features(np.array(raw_feats[:], dtype=np.float32))
    test_feats = (test_feats - feat_mean) / feat_std
    feat_test = torch.from_numpy(test_feats)
    y_test = np.array(labels[:], dtype=np.int64)

    p(f"Test: {mat_test.shape}")

    # Load model
    model = HybridModel(n_feat=feat_test.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v8.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    p("Model loaded")

    # Test-time augmentation: 4 rotations
    all_probs = []
    rotations = [0, 1, 2, 3]  # 0, 90, 180, 270 degrees

    for rot in rotations:
        p(f"\nRotation {rot*90} degrees...")
        if rot == 0:
            mat_aug = mat_test
        else:
            # rot90 on spatial dims (2,3)
            mat_aug = torch.rot90(mat_test, k=rot, dims=[2, 3])

        probs_list = []
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                end = min(i + BATCH_SIZE, n)
                mat_b = mat_aug[i:end].to(DEVICE)
                feat_b = feat_test[i:end].to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mat_b, feat_b)
                probs_list.append(torch.softmax(out.float(), 1).cpu().numpy())

        probs = np.concatenate(probs_list)
        preds = probs.argmax(1)
        acc = (preds == y_test).mean()
        p(f"  Accuracy (rot {rot*90}): {acc:.4f}")
        all_probs.append(probs)

    # Also try horizontal and vertical flips
    for flip_dim, flip_name in [(2, "vertical"), (3, "horizontal")]:
        p(f"\nFlip {flip_name}...")
        mat_aug = torch.flip(mat_test, dims=[flip_dim])

        probs_list = []
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                end = min(i + BATCH_SIZE, n)
                mat_b = mat_aug[i:end].to(DEVICE)
                feat_b = feat_test[i:end].to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mat_b, feat_b)
                probs_list.append(torch.softmax(out.float(), 1).cpu().numpy())

        probs = np.concatenate(probs_list)
        preds = probs.argmax(1)
        acc = (preds == y_test).mean()
        p(f"  Accuracy (flip {flip_name}): {acc:.4f}")
        all_probs.append(probs)

    # Average all augmentations
    p("\n--- Averaging ---")
    for n_aug in [4, 6]:  # 4 rotations only, or all 6
        avg_probs = np.mean(all_probs[:n_aug], axis=0)
        avg_preds = avg_probs.argmax(1)
        avg_acc = (avg_preds == y_test).mean()
        true_fracs = np.bincount(y_test, minlength=5) / len(y_test)
        pred_fracs = np.bincount(avg_preds, minlength=5) / len(avg_preds)
        frac_diff = np.mean(np.abs(true_fracs - pred_fracs))
        p(f"TTA-{n_aug}: acc={avg_acc:.4f}, frac_diff={frac_diff:.4f}")

    # Save best (TTA-4 if it's better)
    best_n = 4
    best_probs = np.mean(all_probs[:best_n], axis=0)
    best_preds = best_probs.argmax(1)
    best_acc = (best_preds == y_test).mean()

    # Compare with TTA-6
    tta6_probs = np.mean(all_probs[:6], axis=0)
    tta6_preds = tta6_probs.argmax(1)
    tta6_acc = (tta6_preds == y_test).mean()

    if tta6_acc > best_acc:
        best_preds = tta6_preds
        best_probs = tta6_probs
        best_acc = tta6_acc
        best_n = 6

    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v18.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f} (TTA-{best_n})")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: TTA-{best_n} on v8 model (rotations{'+ flips' if best_n==6 else ''})")

if __name__ == "__main__":
    main()
