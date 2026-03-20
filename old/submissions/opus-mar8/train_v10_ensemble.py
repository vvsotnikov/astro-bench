"""v10: Ensemble CNN (v8) + HGB predictions.
Uses probability averaging between CNN and HGB, sweeps mixing weights."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

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
        E**2,
        Ne * Nmu,
        (Ne - Nmu)**2,
    ]
    return np.stack(feats, axis=1).astype(np.float32)


def main():
    t0 = time.time()

    # Load CNN probabilities from v8
    p("Loading CNN probabilities from v8...")
    cnn_probs = np.load(f"{OUT_DIR}/probs_v8.npy")  # (119027, 5)
    p(f"  CNN probs shape: {cnn_probs.shape}")

    # Train HGB and get probabilities
    p("Loading training data for HGB...")
    raw_feats = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')

    # Use all data (subsample 3M for speed)
    rng = np.random.RandomState(42)
    n = len(labels)
    idx = rng.choice(n, 3000000, replace=False)
    idx.sort()

    # Load in chunks
    chunk = 500000
    feat_list, label_list = [], []
    for i in range(0, len(idx), chunk):
        batch = idx[i:i+chunk]
        feat_list.append(np.array(raw_feats[batch], dtype=np.float32))
        label_list.append(np.array(labels[batch], dtype=np.int8))
        p(f"  Loaded {min(i+chunk, len(idx))}/{len(idx)}")

    X_train = engineer_features(np.concatenate(feat_list))
    y_train = np.concatenate(label_list)

    # Test features
    raw_test = np.array(np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r'), dtype=np.float32)
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r'))
    X_test = engineer_features(raw_test)

    p(f"Training HGB (n_train={len(y_train)})...")
    hgb = HistGradientBoostingClassifier(
        max_iter=1000,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=0.1,
        random_state=42,
        verbose=0,
    )
    hgb.fit(X_train, y_train)
    hgb_probs = hgb.predict_proba(X_test)  # (119027, 5)
    hgb_preds = hgb.predict(X_test)
    hgb_acc = (hgb_preds == y_test).mean()
    p(f"  HGB accuracy: {hgb_acc:.4f}")
    np.save(f"{OUT_DIR}/probs_hgb.npy", hgb_probs)

    # Try different mixing weights
    p("\nEnsemble sweep (CNN weight):")
    best_acc = 0
    best_w = 0
    for w in np.arange(0.0, 1.05, 0.05):
        mixed = w * cnn_probs + (1 - w) * hgb_probs
        preds = mixed.argmax(1)
        acc = (preds == y_test).mean()
        p(f"  w={w:.2f}: acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_preds = preds

    p(f"\nBest: w={best_w:.2f}, acc={best_acc:.4f}")

    # Save best ensemble predictions
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))

    elapsed = time.time() - t0
    p(f"Done in {elapsed/60:.1f}m")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: Ensemble CNN(v8)*{best_w:.2f} + HGB*{1-best_w:.2f}, 3M HGB train")

if __name__ == "__main__":
    main()
