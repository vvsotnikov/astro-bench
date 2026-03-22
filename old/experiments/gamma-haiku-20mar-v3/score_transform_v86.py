#!/usr/bin/env python3
"""v86: v36 with sigmoid stretching - amplify confidence"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

preds_v36 = np.load('predictions_v36.npz')['gamma_scores']

# Sigmoid stretching: map [0,1] -> [0,1] with increased slope
# Use 4*sigmoid(8*(x-0.5)) to get steeper curve
x = 4.0 * (preds_v36 - 0.5)  # Map [0,1] to [-2, 2]
gamma_scores = 1.0 / (1.0 + np.exp(-x))  # Sigmoid

np.savez_compressed('predictions_v86.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v86: v36 with sigmoid stretching")
print(f"Metric: {metric}")
