"""Physics-informed approach: SVM on key features."""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sys

# Load test data
test_features = np.load("data/gamma_test/features.npy")[:]
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

# Load training data
train_features = np.load("data/gamma_train/features.npy", mmap_mode='r')[:]
train_labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode='r')[:]

print(f"Training set: {len(train_labels)} events")
print(f"Test set: {len(test_labels)} events")

# Extract key features
# Features: 0=E, 1=Ze, 2=Az, 3=Ne, 4=Nmu
E_train = train_features[:, 0]
Ze_train = train_features[:, 1]
Ne_train = train_features[:, 3]
Nmu_train = train_features[:, 4]

# Create engineered features (log space)
X_train = np.stack([
    Ne_train,              # electron number
    Nmu_train,             # muon number
    Ne_train - Nmu_train,  # difference (strongest discriminant)
    Ne_train / (Nmu_train + 0.1),  # ratio
    E_train,               # energy
    Ze_train,              # zenith
], axis=1)

E_test = test_features[:, 0]
Ze_test = test_features[:, 1]
Ne_test = test_features[:, 3]
Nmu_test = test_features[:, 4]

X_test = np.stack([
    Ne_test,
    Nmu_test,
    Ne_test - Nmu_test,
    Ne_test / (Nmu_test + 0.1),
    E_test,
    Ze_test,
], axis=1)

print(f"Feature matrix: {X_train.shape}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training SVM...")
svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, verbose=1)
svm.fit(X_train_scaled, train_labels)

print("Getting predictions...")
test_probs = svm.decision_function(X_test_scaled)

# Normalize to [0, 1]
test_probs_norm = (test_probs - test_probs.min()) / (test_probs.max() - test_probs.min() + 1e-8)

# Save
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v4.npz",
         gamma_scores=test_probs_norm)

# Evaluate
is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

survival = compute_survival_75(test_probs_norm)
print(f"Test survival @ 75% gamma eff: {survival:.4e}")

print(f"\n---")
print(f"metric: {survival:.4e}")
print(f"description: SVM (RBF) on engineered features (Ne, Nmu, ratio, diff)")
