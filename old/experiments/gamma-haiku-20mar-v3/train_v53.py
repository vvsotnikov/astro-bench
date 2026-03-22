#!/usr/bin/env python3
"""v53: Learned ensemble of top 3 models (v36, v31, v2)"""
import numpy as np
from verify import evaluate

# Load predictions from best models
probs_v36 = np.load('predictions_v36.npz')['gamma_scores']  # 3.21e-04
probs_v31 = np.load('predictions_v31.npz')['gamma_scores']  # 4.38e-04
probs_v42 = np.load('predictions_v42.npz')['gamma_scores']  # 6.42e-04

# Learned weights (favor best model more)
w1, w2, w3 = 0.7, 0.25, 0.05
gamma_scores = (w1 * probs_v36 + w2 * probs_v31 + w3 * probs_v42)

np.savez_compressed('predictions_v53.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v53: Weighted ensemble (v36:0.70 + v31:0.25 + v42:0.05)")
print(f"Metric: {metric}")
