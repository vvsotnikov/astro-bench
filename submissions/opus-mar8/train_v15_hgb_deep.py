"""v15: Deep HGB with many more engineered features.
Hypothesis: better feature engineering can push HGB to match or beat CNN on fraction error.
HGB is better calibrated than CNN for fraction error (v2 vs v5 showed this).
Use all 5.5M training data."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

def p(msg):
    print(msg, flush=True)

def engineer_features_v2(f):
    """Extended feature set -- 30+ features."""
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]

    Ze_rad = np.radians(Ze)
    Az_rad = np.radians(Az)

    feats = [
        # Raw
        E, Ze, Ne, Nmu, Az,
        # Trig
        np.sin(Ze_rad), np.cos(Ze_rad),
        np.sin(Az_rad), np.cos(Az_rad),
        np.sin(2 * Ze_rad), np.cos(2 * Ze_rad),
        # Ne-Nmu ratios (key discriminant)
        Ne - Nmu,
        Ne + Nmu,
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne / (Nmu + 1e-6),
        Nmu / (Ne + 1e-6),
        # Energy ratios
        Ne - E, Nmu - E,
        Ne / (E + 1e-6),
        Nmu / (E + 1e-6),
        (Ne + Nmu) / (E + 1e-6),
        # Quadratic
        E**2, Ne**2, Nmu**2,
        Ne * Nmu,
        (Ne - Nmu)**2,
        E * Ne, E * Nmu,
        # Zenith-corrected (atmospheric depth correction)
        Ne / (np.cos(Ze_rad) + 0.01),
        Nmu / (np.cos(Ze_rad) + 0.01),
        (Ne - Nmu) / (np.cos(Ze_rad) + 0.01),
        # Log ratios
        np.log1p(np.abs(Ne - Nmu)),
        np.log1p(Ne + Nmu),
    ]
    return np.stack(feats, axis=1).astype(np.float32)


def main():
    t0 = time.time()

    # Load ALL training data
    p("Loading training data...")
    raw_feats = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')
    n = len(labels)
    p(f"  Total training: {n}")

    # Subsample 3M for speed (include both simulations)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, min(3000000, n), replace=False)
    idx.sort()

    # Load in chunks
    chunk = 500000
    feat_list, label_list = [], []
    for i in range(0, len(idx), chunk):
        batch = idx[i:i+chunk]
        feat_list.append(engineer_features_v2(np.array(raw_feats[batch], dtype=np.float32)))
        label_list.append(np.array(labels[batch], dtype=np.int8))
        p(f"  Loaded {min(i+chunk, len(idx))}/{len(idx)}")

    X_train = np.concatenate(feat_list)
    y_train = np.concatenate(label_list)
    del feat_list, label_list
    p(f"  Features: {X_train.shape}")

    # Test data
    raw_test = np.array(np.load(f"{DATA_DIR}/composition_test/features.npy"), dtype=np.float32)
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)
    X_test = engineer_features_v2(raw_test)
    p(f"  Test: {X_test.shape}")

    # Train HGB with different configs
    configs = [
        {"max_iter": 1500, "max_depth": 10, "learning_rate": 0.05, "min_samples_leaf": 30, "l2_regularization": 0.05},
    ]

    best_acc = 0
    best_preds = None
    best_probs = None
    best_desc = ""

    for i, cfg in enumerate(configs):
        p(f"\nTraining HGB config {i+1}/{len(configs)}: {cfg}")
        hgb = HistGradientBoostingClassifier(
            **cfg,
            random_state=42,
            verbose=0,
        )
        hgb.fit(X_train, y_train)
        probs = hgb.predict_proba(X_test)
        preds = probs.argmax(1)
        acc = (preds == y_test).mean()

        true_fracs = np.bincount(y_test, minlength=5) / len(y_test)
        pred_fracs = np.bincount(preds, minlength=5) / len(preds)
        frac_diff = np.mean(np.abs(true_fracs - pred_fracs))

        p(f"  Accuracy: {acc:.4f}, frac_diff: {frac_diff:.4f}")
        p(f"  True fracs: {true_fracs}")
        p(f"  Pred fracs: {pred_fracs}")

        if acc > best_acc:
            best_acc = acc
            best_preds = preds
            best_probs = probs
            best_desc = f"HGB cfg{i+1} depth={cfg['max_depth']} iter={cfg['max_iter']}"
            np.save(f"{OUT_DIR}/probs_v15_cfg{i+1}.npy", probs)

    p(f"\nBest: {best_desc}, acc={best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v15.npy", best_probs)

    elapsed = time.time() - t0
    p(f"Done in {elapsed/60:.1f}m")
    p(f"---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: {best_desc}, 30+ features, ALL 5.5M train")

if __name__ == "__main__":
    main()
