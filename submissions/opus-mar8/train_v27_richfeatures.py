"""v27: HGB with rich matrix-derived features.
Extract statistical features from 16x16x2 matrices and combine with scalar features.
This creates a fundamentally different model from CNN (different inductive bias)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

def p(msg):
    print(msg, flush=True)

def extract_matrix_features(matrices, chunk_size=200000):
    """Extract statistical features from 16x16x2 matrices."""
    n = len(matrices)
    all_feats = []

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        m = np.array(matrices[i:end], dtype=np.float32)  # (B, 16, 16, 2)

        ch0 = m[:, :, :, 0]  # electron channel
        ch1 = m[:, :, :, 1]  # muon channel

        feats = []
        for ch, name in [(ch0, 'e'), (ch1, 'mu')]:
            flat = ch.reshape(len(ch), -1)
            feats.extend([
                flat.sum(axis=1),           # total sum
                flat.max(axis=1),           # max value
                (flat > 0).sum(axis=1),     # nonzero count
                flat.mean(axis=1),          # mean
                flat.std(axis=1),           # std
                np.percentile(flat, 90, axis=1),  # 90th percentile
                np.percentile(flat, 99, axis=1),  # 99th percentile
            ])

            # log1p versions
            log_flat = np.log1p(flat)
            feats.extend([
                log_flat.sum(axis=1),
                log_flat.max(axis=1),
                log_flat.mean(axis=1),
                log_flat.std(axis=1),
            ])

            # Spatial features: center of mass
            x_coords = np.arange(16).reshape(1, 1, 16)
            y_coords = np.arange(16).reshape(1, 16, 1)
            total = ch.sum(axis=(1, 2)) + 1e-10
            cx = (ch * x_coords).sum(axis=(1, 2)) / total
            cy = (ch * y_coords).sum(axis=(1, 2)) / total
            feats.extend([cx, cy])

            # Radial features (distance from center)
            r = np.sqrt((x_coords - 7.5)**2 + (y_coords - 7.5)**2)
            r_mean = (ch * r).sum(axis=(1, 2)) / total
            feats.append(r_mean)

        # Cross-channel features
        ratio = np.log1p(ch0.reshape(len(ch0), -1)).sum(axis=1) / (np.log1p(ch1.reshape(len(ch1), -1)).sum(axis=1) + 1e-6)
        feats.append(ratio)

        batch_feats = np.stack(feats, axis=1)
        all_feats.append(batch_feats)
        if (i // chunk_size) % 5 == 0:
            p(f"  matrix features: {end}/{n}")

    return np.concatenate(all_feats, axis=0)


def engineer_scalar_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    return np.stack([
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu, (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
        Ne**2, Nmu**2, (Ne - Nmu)**2,
    ], axis=1).astype(np.float32)


def load_features(split, max_train=3000000):
    p(f"Loading {split}...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    if split == "train" and max_train and max_train < n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_train, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    # Scalar features
    feat_chunks = []
    for i in range(0, len(idx), 500000):
        end = min(i + 500000, len(idx))
        batch_idx = idx[i:end]
        feat_chunks.append(engineer_scalar_features(np.array(raw_feats[batch_idx], dtype=np.float32)))
    scalar = np.concatenate(feat_chunks)
    p(f"  Scalar features: {scalar.shape}")

    # Matrix features
    # Need to load in the order of idx
    mat_feat_chunks = []
    chunk = 200000
    for i in range(0, len(idx), chunk):
        end = min(i + chunk, len(idx))
        batch_idx = idx[i:end]
        m = np.array(matrices[batch_idx], dtype=np.float32)

        ch0 = m[:, :, :, 0]
        ch1 = m[:, :, :, 1]

        feats = []
        for ch in [ch0, ch1]:
            flat = ch.reshape(len(ch), -1)
            feats.extend([
                flat.sum(axis=1),
                flat.max(axis=1),
                (flat > 0).sum(axis=1).astype(np.float32),
                flat.mean(axis=1),
                flat.std(axis=1),
            ])
            log_flat = np.log1p(flat)
            feats.extend([
                log_flat.sum(axis=1),
                log_flat.max(axis=1),
                log_flat.mean(axis=1),
                log_flat.std(axis=1),
            ])
            # Spatial
            x_coords = np.arange(16).reshape(1, 1, 16)
            y_coords = np.arange(16).reshape(1, 16, 1)
            total = ch.sum(axis=(1, 2)) + 1e-10
            cx = (ch * x_coords).sum(axis=(1, 2)) / total
            cy = (ch * y_coords).sum(axis=(1, 2)) / total
            feats.extend([cx, cy])
            r = np.sqrt((x_coords.squeeze() - 7.5)**2 + (y_coords.squeeze() - 7.5)**2)
            r_mean = (ch * r).sum(axis=(1, 2)) / total
            feats.append(r_mean)

        # Cross-channel
        ratio = (np.log1p(ch0.reshape(len(ch0), -1)).sum(axis=1) /
                 (np.log1p(ch1.reshape(len(ch1), -1)).sum(axis=1) + 1e-6))
        feats.append(ratio)

        batch_feats = np.stack(feats, axis=1).astype(np.float32)
        mat_feat_chunks.append(batch_feats)
        if (i // chunk) % 5 == 0:
            p(f"  mat features: {end}/{len(idx)}")

    mat_feats = np.concatenate(mat_feat_chunks)
    p(f"  Matrix features: {mat_feats.shape}")

    # Combine
    X = np.concatenate([scalar, mat_feats], axis=1)
    y = np.array(labels[:], dtype=int)[idx]

    p(f"  Combined: {X.shape}, labels: {np.bincount(y, minlength=5)}")
    return X, y


def main():
    t0 = time.time()

    X_train, y_train = load_features("train", max_train=3000000)
    X_test, y_test = load_features("test")

    n_features = X_train.shape[1]
    p(f"\nTotal features: {n_features}")
    p(f"Training HGB with 1000 iterations...")

    clf = HistGradientBoostingClassifier(
        max_iter=1000,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=0.1,
        max_bins=255,
        random_state=42,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    preds = probs.argmax(1)
    acc = (preds == y_test).mean()
    p(f"\nTest accuracy: {acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions_v27.npz", predictions=preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v27.npy", probs)

    elapsed = time.time() - t0
    p(f"Done in {elapsed/60:.1f}m")
    p(f"---")
    p(f"metric: {acc:.4f}")
    p(f"description: HGB 1000iter, 16 scalar + matrix stats ({n_features} total), 3M train")


if __name__ == "__main__":
    main()
