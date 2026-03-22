"""Test different ensemble weightings once all seed predictions are available."""
import numpy as np
from verify import evaluate
import os
import time

# Wait for all seed predictions
seeds = [3, 16, 17, 18, 19]
max_wait = 600  # 10 minutes
start = time.time()

while time.time() - start < max_wait:
    all_ready = all(os.path.exists(f"predictions_v{s}.npz") for s in seeds)
    if all_ready:
        print("All seed predictions available!")
        break
    time.sleep(10)
else:
    print("Timeout waiting for predictions")
    exit(1)

# Load all predictions
preds = {}
for s in seeds:
    preds[s] = np.load(f"predictions_v{s}.npz")["gamma_scores"]

# Test multiple weight schemes
schemes = {
    "equal_5seed": {3: 0.2, 16: 0.2, 17: 0.2, 18: 0.2, 19: 0.2},
    "v3_heavy": {3: 0.4, 16: 0.15, 17: 0.15, 18: 0.15, 19: 0.15},
    "v3_dominant": {3: 0.6, 16: 0.1, 17: 0.1, 18: 0.1, 19: 0.1},
    "equal_3best": {3: 0.33, 16: 0.33, 17: 0.34, 18: 0, 19: 0},  # Exclude v18, v19
}

for name, weights in schemes.items():
    ensemble = sum(w * preds[s] for s, w in weights.items() if w > 0)
    np.savez(f"predictions_v_ensemble_{name}.npz", gamma_scores=ensemble)
    evaluate(ensemble, f"Ensemble: {name} ({', '.join(f'v{s}:{w:.2f}' for s,w in weights.items() if w > 0)})")
