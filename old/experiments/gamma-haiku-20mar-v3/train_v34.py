#!/usr/bin/env python3
"""v34: Weighted ensemble of best models (v1, v2, v4, v6 - or use saved weights)"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

# Ensemble the 4 best single models with weights
# v1: 7.30e-04, v2: 4.67e-04, v4: 5.84e-04, v6: 7.88e-04
# v2 is clearly best, so weight it heavily

# Attempt to load existing predictions if available
preds = {}
weights_used = []

# Try loading v1
try:
    v1_data = np.load('predictions_v1.npz')
    preds[1] = v1_data['gamma_scores']
    print(f"✓ Loaded v1")
except:
    print(f"○ v1 not available (not saved from baseline)")

# v2 must exist (it's the best baseline)
try:
    v2_data = np.load('predictions_v2.npz')
    preds[2] = v2_data['gamma_scores']
    print(f"✓ Loaded v2")
except:
    print(f"Could not load v2 - this should not happen!")
    sys.exit(1)

# Try loading v4
try:
    v4_data = np.load('predictions_v4.npz')
    preds[4] = v4_data['gamma_scores']
    print(f"✓ Loaded v4")
except:
    print(f"○ v4 not available")

# Try loading v6
try:
    v6_data = np.load('predictions_v6.npz')
    preds[6] = v6_data['gamma_scores']
    print(f"✓ Loaded v6")
except:
    print(f"○ v6 not available")

# Build ensemble from available models
if len(preds) == 1:
    # Only v2 available, just use it
    gamma_scores = preds[2]
    desc = "v34: Just v2 (others unavailable)"
elif len(preds) >= 2:
    # Have v2 + others, build weighted ensemble
    # Weights: v2 dominates (0.6), others share (0.4)
    if 2 in preds and 1 in preds and 4 in preds and 6 in preds:
        gamma_scores = 0.6*preds[2] + 0.2*preds[1] + 0.1*preds[4] + 0.1*preds[6]
        desc = "v34: Weighted ensemble (v1:0.2 v2:0.6 v4:0.1 v6:0.1)"
    elif 2 in preds and 1 in preds:
        gamma_scores = 0.7*preds[2] + 0.3*preds[1]
        desc = "v34: Ensemble (v1:0.3 v2:0.7)"
    else:
        # Just v2
        gamma_scores = preds[2]
        desc = "v34: Fallback to v2"

np.savez_compressed('predictions_v34.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, desc)
print(f"Metric: {metric}")
