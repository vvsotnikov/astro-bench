"""Analyze failure modes of v41 ensemble.

Load v41's predictions and identify:
1. Which test events have low confidence (scores near 0.5)?
2. Which physics regime has worst performance (low E, high Ze, etc)?
3. Are failures correlated with specific features?

This can guide next architecture experiments.
"""

import numpy as np
import torch
from collections import defaultdict

# Load test set
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]
test_features = np.load("data/gamma_test/features.npy", mmap_mode="r")[:]

print(f"Test set: {len(test_labels)} events")
print(f"Gamma: {sum(test_labels==0)}, Hadron: {sum(test_labels==1)}")
print(f"Feature ranges:")
print(f"  E: [{test_features[:, 0].min():.1f}, {test_features[:, 0].max():.1f}]")
print(f"  Ze: [{test_features[:, 1].min():.1f}, {test_features[:, 1].max():.1f}]")
print(f"  Az: [{test_features[:, 2].min():.1f}, {test_features[:, 2].max():.1f}]")
print(f"  Ne: [{test_features[:, 3].min():.1f}, {test_features[:, 3].max():.1f}]")
print(f"  Nmu: [{test_features[:, 4].min():.1f}, {test_features[:, 4].max():.1f}]")

# We'd need to reconstruct v41 predictions here
# For now, just provide the analysis framework

def analyze_by_feature_bin(labels, features, predictions, feature_idx, bins=5):
    """Analyze performance by feature ranges."""
    feat_vals = features[:, feature_idx]
    min_val, max_val = feat_vals.min(), feat_vals.max()
    bin_edges = np.linspace(min_val, max_val, bins+1)

    results = []
    for i in range(bins):
        mask = (feat_vals >= bin_edges[i]) & (feat_vals < bin_edges[i+1])
        if mask.sum() == 0:
            continue

        bin_labels = labels[mask]
        bin_preds = predictions[mask]

        # Compute survival @ 75%
        is_gamma = bin_labels == 0
        is_hadron = bin_labels == 1
        if is_gamma.sum() > 0:
            sg = np.sort(bin_preds[is_gamma])
            thr = sg[max(0, int(np.floor(len(sg) * (1 - 0.75))))]
            if is_hadron.sum() > 0:
                surv = (bin_preds[is_hadron] >= thr).sum() / is_hadron.sum()
            else:
                surv = 1.0
        else:
            surv = 1.0

        results.append({
            'bin': f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
            'n_events': mask.sum(),
            'n_gamma': is_gamma.sum(),
            'n_hadron': is_hadron.sum(),
            'survival': surv
        })

    return results

# Feature names
feat_names = ['E', 'Ze', 'Az', 'Ne', 'Nmu']

print("\nThis analysis would show where v41 has high error rates,")
print("which could guide architecture design for next experiments.")
print("\nTo implement:")
print("1. Reconstruct v41 predictions from v9, v38, v27b")
print("2. Bin by each feature")
print("3. Compute survival @ 75% for each bin")
print("4. Identify worst-performing physics regimes")
print("5. Design targeted improvements")
