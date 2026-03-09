"""Calibrate DNN scores using isotonic regression."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import sys

# Load test predictions from previous run
dnn_scores = np.load("/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar9/predictions.npz")["gamma_scores"]

# Load test features and labels
test_features = np.load("/home/vladimir/cursor_projects/astro-agents/data/gamma_test/features.npy")[:]
test_labels = np.load("/home/vladimir/cursor_projects/astro-agents/data/gamma_test/labels_gamma.npy")[:]

# Also load training data to get validation fold for calibration
train_features = np.load("/home/vladimir/cursor_projects/astro-agents/data/gamma_train/features.npy", mmap_mode='r')
train_matrices = np.load("/home/vladimir/cursor_projects/astro-agents/data/gamma_train/matrices.npy", mmap_mode='r')
train_labels = np.load("/home/vladimir/cursor_projects/astro-agents/data/gamma_train/labels_gamma.npy", mmap_mode='r')

N = len(train_labels)
n_test = int(0.2 * N)
n_train = N - n_test

print(f"Total training events: {N}")
print(f"Train subset: {n_train}, validation subset: {n_test}")

# Load the DNN model from previous run and get validation scores
import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(517, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Try to load the best model
model_path = "/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar9/model_best_v2.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    model = DNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get validation scores
    val_matrices = train_matrices[n_train:][:].astype(np.float32)
    val_features = train_features[n_train:][:].astype(np.float32)
    val_labels = train_labels[n_train:][:]

    val_matrices_flat = val_matrices.reshape(len(val_matrices), -1)
    val_input = np.concatenate([val_matrices_flat, val_features], axis=1)

    val_tensor = torch.from_numpy(val_input).to(device)
    with torch.no_grad():
        val_logits = model(val_tensor).cpu().numpy()
    val_scores = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()[:, 0]

    print(f"Validation scores shape: {val_scores.shape}")
    print(f"Validation scores range: [{val_scores.min():.4f}, {val_scores.max():.4f}]")

    # Fit isotonic regression on validation set
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(val_scores, val_labels == 0)  # True if gamma

    # Apply to test predictions
    calibrated_scores = iso_reg.predict(dnn_scores)

    print(f"Calibrated scores range: [{calibrated_scores.min():.4f}, {calibrated_scores.max():.4f}]")

    # Evaluate
    is_gamma = test_labels == 0
    is_hadron = test_labels == 1

    def compute_survival(scores):
        sg = np.sort(scores[is_gamma])
        ng = len(sg)
        thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
        n_surv = (scores[is_hadron] >= thr).sum()
        return n_surv / is_hadron.sum()

    orig_surv = compute_survival(dnn_scores)
    calib_surv = compute_survival(calibrated_scores)

    print(f"\nOriginal survival @ 75% gamma eff: {orig_surv:.4e}")
    print(f"Calibrated survival @ 75% gamma eff: {calib_surv:.4e}")

    # Save
    np.savez("/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar9-v2/predictions_v1_calibrate.npz",
             gamma_scores=calibrated_scores)

    print(f"\n---")
    print(f"metric: {calib_surv:.4e}")
    print(f"description: Isotonic regression calibration of DNN scores")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
