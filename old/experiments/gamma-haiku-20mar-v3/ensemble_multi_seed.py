#!/usr/bin/env python3
"""v79: Ensemble multiple seeds - average v36 (seed 42) + v71 (seed 777) + v72 (seed 999)"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

# Load predictions from different seeds
# v36: seed 42 (training default)
# v71: seed 777 (good model)
# v72: seed 999 (good model)

try:
    preds_v36 = np.load('predictions_v36.npz')['gamma_scores']
    print(f"Loaded v36: {len(preds_v36)} predictions")
except Exception as e:
    print(f"Error loading v36: {e}")
    preds_v36 = None

try:
    preds_v71 = np.load('predictions_v71.npz')['gamma_scores']
    print(f"Loaded v71: {len(preds_v71)} predictions")
except Exception as e:
    print(f"Error loading v71: {e}")
    preds_v71 = None

try:
    preds_v72 = np.load('predictions_v72.npz')['gamma_scores']
    print(f"Loaded v72: {len(preds_v72)} predictions")
except Exception as e:
    print(f"Error loading v72: {e}")
    preds_v72 = None

# Simple average ensemble
if preds_v36 is not None and preds_v71 is not None and preds_v72 is not None:
    gamma_scores = (preds_v36 + preds_v71 + preds_v72) / 3.0
    np.savez_compressed('predictions_v79.npz', gamma_scores=gamma_scores)
    metric = evaluate(gamma_scores, "v79: Ensemble v36 + v71 + v72 (seeds 42/777/999)")
    print(f"Metric: {metric}")
else:
    print("Not all prediction files found")
