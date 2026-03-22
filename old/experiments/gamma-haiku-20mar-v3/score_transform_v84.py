#!/usr/bin/env python3
"""v84: v36 with inverse (1 - scores) - reverse predictions"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

preds_v36 = np.load('predictions_v36.npz')['gamma_scores']

# Reverse predictions to see if class is inverted
gamma_scores = 1.0 - preds_v36

np.savez_compressed('predictions_v84.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v84: v36 reversed (1 - scores)")
print(f"Metric: {metric}")
