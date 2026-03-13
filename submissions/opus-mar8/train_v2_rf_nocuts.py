"""v2: RandomForest on all training data (no quality cuts), more features."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

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
        E**2,
        Ne * Nmu,
        (Ne - Nmu)**2,
    ]
    return np.stack(feats, axis=1).astype(np.float32)

def main():
    t0 = time.time()

    print("Loading training data (no cuts)...")
    raw_feats = np.array(np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r'), dtype=np.float32)
    labels = np.array(np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r'))

    # Subsample for speed (2M events)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(labels), 2000000, replace=False)
    raw_feats = raw_feats[idx]
    labels = labels[idx]

    X_train = engineer_features(raw_feats)
    y_train = labels

    print("Loading test data...")
    raw_test = np.array(np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r'), dtype=np.float32)
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r'))
    X_test = engineer_features(raw_test)

    # Train HistGradientBoosting (much faster than RF on large data)
    print(f"Training HistGradientBoosting (n_train={len(y_train)}, n_test={len(y_test)})...")
    clf = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=0.1,
        random_state=42,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"Test accuracy: {acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"---")
    print(f"metric: {acc:.4f}")
    print(f"description: HGB 500 iter, 2M train, no cuts, 16 eng features")

if __name__ == "__main__":
    main()
