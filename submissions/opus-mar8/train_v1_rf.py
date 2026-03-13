"""v1: RandomForest baseline on engineered features."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

def engineer_features(f):
    """Create engineered features from raw 5 features [E, Ze, Az, Ne, Nmu]."""
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    feats = [
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)),
        np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)),
        np.cos(np.radians(Az)),
        Ne - Nmu,           # log(Ne/Nmu) -- strongest discriminant
        Ne + Nmu,           # log(Ne*Nmu)
        (Ne - Nmu) / (Ne + Nmu + 1e-6),  # normalized ratio
        Ne - E,             # electron excess relative to energy
        Nmu - E,            # muon excess relative to energy
        E**2,               # energy squared
        Ne * Nmu,           # interaction
        (Ne - Nmu)**2,      # squared ratio
    ]
    return np.stack(feats, axis=1).astype(np.float32)

def main():
    t0 = time.time()

    # Load training data
    print("Loading training data...")
    raw_feats = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')

    # Apply quality cuts to training data (match test set)
    print("Applying quality cuts...")
    feats_all = np.array(raw_feats, dtype=np.float32)
    labels_all = np.array(labels, dtype=np.int8)

    Ze = feats_all[:, 1]
    Ne = feats_all[:, 3]
    # We don't have Age in features, so just apply Ze and Ne cuts
    mask = (Ze < 30) & (Ne > 4.8)
    feats_cut = feats_all[mask]
    labels_cut = labels_all[mask]
    print(f"  After cuts: {len(labels_cut)} / {len(labels_all)} events ({100*len(labels_cut)/len(labels_all):.1f}%)")

    X_train = engineer_features(feats_cut)
    y_train = labels_cut

    # Load test data
    print("Loading test data...")
    raw_test = np.array(np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r'), dtype=np.float32)
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r'))
    X_test = engineer_features(raw_test)

    # Train RF
    print(f"Training RandomForest (n_train={len(y_train)}, n_test={len(y_test)})...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    rf.fit(X_train, y_train)

    # Predict
    preds = rf.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"Test accuracy: {acc:.4f}")

    # Save predictions
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"---")
    print(f"metric: {acc:.4f}")
    print(f"description: RF 500 trees, 16 engineered features, quality cuts on train")

if __name__ == "__main__":
    main()
