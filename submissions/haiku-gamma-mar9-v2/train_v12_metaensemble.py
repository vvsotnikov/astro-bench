"""Learn ensemble weights using validation set."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load training data for validation
train_features = np.load("data/gamma_train/features.npy", mmap_mode='r')
train_matrices = np.load("data/gamma_train/matrices.npy", mmap_mode='r')
train_labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode='r')

N = len(train_labels)
n_train = int(0.8 * N)
n_val = N - n_train

val_start = n_train
val_end = N

print(f"Validation set: {n_val} events")

# Load models to get validation scores
device = torch.device("cuda:0")

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(517, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512 // 2),
            nn.BatchNorm1d(512 // 2),
            nn.ReLU(),
            nn.Linear(512 // 2, 2),
        )

    def forward(self, x):
        return self.net(x)

class RegDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(517, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512 // 2),
            nn.BatchNorm1d(512 // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512 // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# Load models
model_v2 = DNN().to(device)
model_v2.load_state_dict(torch.load("submissions/haiku-gamma-mar9-v2/model_best_v2.pt"))
model_v2.eval()

model_v3 = RegDNN().to(device)
model_v3.load_state_dict(torch.load("submissions/haiku-gamma-mar9-v2/model_best_v3.pt"))
model_v3.eval()

# Compute normalization stats from training
train_matrices_flat = train_matrices[:n_train].reshape(n_train, -1).astype(np.float32)
train_features_part = train_features[:n_train].astype(np.float32)
train_X = np.concatenate([train_matrices_flat, train_features_part], axis=1)

mean = train_X.mean(axis=0)
std = train_X.std(axis=0)
std[std == 0] = 1.0

# Get validation scores
val_matrices = train_matrices[val_start:val_end].reshape(n_val, -1).astype(np.float32)
val_features = train_features[val_start:val_end].astype(np.float32)
val_X = np.concatenate([val_matrices, val_features], axis=1)
val_X_norm = (val_X - mean) / (std + 1e-8)

val_labels = train_labels[val_start:val_end]

# Batch inference
batch_size = 8192
v2_val_scores = []
v3_val_scores = []

with torch.no_grad():
    for i in range(0, n_val, batch_size):
        end = min(i + batch_size, n_val)
        batch = torch.from_numpy(val_X_norm[i:end]).to(device)

        logits_v2 = model_v2(batch)
        probs_v2 = torch.softmax(logits_v2, dim=1)
        v2_val_scores.append(probs_v2[:, 0].cpu().numpy())

        scores_v3 = model_v3(batch)
        v3_val_scores.append(scores_v3.cpu().numpy())

v2_val = np.concatenate(v2_val_scores)
v3_val = np.concatenate(v3_val_scores)

# Grid search for best weights
is_gamma = val_labels == 0
is_hadron = val_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

# Normalize
v2_norm = (v2_val - v2_val.min()) / (v2_val.max() - v2_val.min() + 1e-8)
v3_norm = (v3_val - v3_val.min()) / (v3_val.max() - v3_val.min() + 1e-8)

print("Optimizing weights on validation set...")
best_surv = 1.0
best_alpha = 0.5

for alpha in np.linspace(0.99, 1.0, 101):
    ensemble = alpha * v2_norm + (1 - alpha) * v3_norm
    surv = compute_survival_75(ensemble)
    if surv < best_surv:
        best_surv = surv
        best_alpha = alpha

print(f"Best on validation: alpha={best_alpha:.3f}, survival={best_surv:.4e}")

# Now apply to test set
test_scores_v2 = np.load("submissions/haiku-gamma-mar9-v2/predictions_v2.npz")["gamma_scores"]
test_scores_v3 = np.load("submissions/haiku-gamma-mar9-v2/predictions_v3.npz")["gamma_scores"]

test_scores_v2_norm = (test_scores_v2 - test_scores_v2.min()) / (test_scores_v2.max() - test_scores_v2.min() + 1e-8)
test_scores_v3_norm = (test_scores_v3 - test_scores_v3.min()) / (test_scores_v3.max() - test_scores_v3.min() + 1e-8)

test_ensemble = best_alpha * test_scores_v2_norm + (1 - best_alpha) * test_scores_v3_norm
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v12.npz",
         gamma_scores=test_ensemble)

test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]
is_gamma_test = test_labels == 0
is_hadron_test = test_labels == 1

def compute_survival_75_test(scores):
    sg = np.sort(scores[is_gamma_test])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron_test] >= thr).sum()
    return n_surv / is_hadron_test.sum()

test_surv = compute_survival_75_test(test_ensemble)
print(f"Test performance: {test_surv:.4e}")

print(f"\n---")
print(f"metric: {test_surv:.4e}")
print(f"description: Weights optimized on validation set (α={best_alpha:.3f})")
