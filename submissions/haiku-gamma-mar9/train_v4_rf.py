"""Random Forest baseline - matches published approach."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time


def compute_survival_at_99(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr_99 = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
    n_hadron_surviving = (scores[is_hadron] >= thr_99).sum()
    survival_99 = n_hadron_surviving / is_hadron.sum()
    return survival_99, thr_99


print("Loading training data...")
X_train = np.load("data/gamma_train/matrices.npy", mmap_mode="r").reshape(
    len(np.load("data/gamma_train/matrices.npy", mmap_mode="r")), -1
)
f_train = np.load("data/gamma_train/features.npy", mmap_mode="r")[:]
y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")[:]

# Flatten matrices and concatenate with features
print(f"Flattening matrices: {X_train.shape[0]} x {X_train.shape[1]}")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_train_full = np.hstack([X_train_flat, f_train])
print(f"Full feature shape: {X_train_full.shape}")

# Subsample for speed (RF is slow on 1.5M samples)
rng = np.random.default_rng(2026)
idx_sub = rng.choice(len(X_train_full), size=100_000, replace=False)
X_sub = X_train_full[idx_sub]
y_sub = y_train[idx_sub]

print(f"Subsampled to {X_sub.shape}")
print(f"Class balance: {(y_sub==0).sum()} gamma, {(y_sub==1).sum()} hadron")

# Train RF
print("\nTraining RF...")
start = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    n_jobs=8,
    class_weight='balanced',
    random_state=2026
)
rf.fit(X_sub, y_sub)
print(f"Training took {time.time()-start:.1f}s")

# Predict on test
print("Loading test data...")
X_test = np.load("data/gamma_test/matrices.npy", mmap_mode="r").reshape(
    len(np.load("data/gamma_test/matrices.npy", mmap_mode="r")), -1
)
f_test = np.load("data/gamma_test/features.npy", mmap_mode="r")[:]
y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")[:]

X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_test_full = np.hstack([X_test_flat, f_test])

print("Predicting...")
start = time.time()
probs = rf.predict_proba(X_test_full)[:, 0]  # P(gamma)
print(f"Prediction took {time.time()-start:.1f}s")

survival, thr = compute_survival_at_99(probs, y_test)
print(f"\nThreshold: {thr:.4f}")
print(f"Test survival @ 99% gamma eff: {survival:.4f}")

np.savez("submissions/haiku-gamma-mar9/predictions_v4_rf.npz",
         gamma_scores=probs)
