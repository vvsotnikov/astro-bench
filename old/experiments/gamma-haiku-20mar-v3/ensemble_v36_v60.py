#!/usr/bin/env python3
"""v79: Ensemble v36 (0.8) + v60 (0.2) - weighted combination of best + label smoothing"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

preds_v36 = np.load('predictions_v36.npz')['gamma_scores']
preds_v60 = np.load('predictions_v60.npz')['gamma_scores']

# Weighted ensemble: v36 dominates
gamma_scores = 0.8 * preds_v36 + 0.2 * preds_v60

np.savez_compressed('predictions_v79.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v79: Ensemble v36 (0.8) + v60 (0.2)")
print(f"Metric: {metric}")
