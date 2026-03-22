#!/usr/bin/env python3
"""v40: Ensemble of v2 (best full) + v39 (muon-only) - test information fusion"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

# Load the two best models' predictions
# v2: 4.67e-04 (best full ResNet)
# v39: muon-only variant (unknown performance)

try:
    v2_data = np.load('predictions_v2.npz')
    v2_scores = v2_data['gamma_scores']
    print(f"✓ Loaded v2: {v2_scores.shape}")
except Exception as e:
    print(f"Error loading v2: {e}")
    sys.exit(1)

# Wait for v39 to complete before continuing
import time
for attempt in range(120):  # Wait up to 10 min
    try:
        v39_data = np.load('predictions_v39.npz')
        v39_scores = v39_data['gamma_scores']
        print(f"✓ Loaded v39: {v39_scores.shape}")
        break
    except:
        if attempt % 12 == 0:
            print(f"Waiting for v39... ({attempt*5}s elapsed)")
        time.sleep(5)
else:
    print("Timeout waiting for v39")
    sys.exit(1)

# Simple average ensemble (v2 dominates since it's better)
gamma_scores = 0.7 * v2_scores + 0.3 * v39_scores

np.savez_compressed('predictions_v40.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v40: Ensemble of v2 (0.7) + v39 muon-only (0.3)")
print(f"Metric: {metric}")
