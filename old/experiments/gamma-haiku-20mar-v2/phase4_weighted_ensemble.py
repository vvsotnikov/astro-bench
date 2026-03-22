"""Phase 4: Test weighted ensemble combinations of existing predictions."""
import numpy as np
from verify import evaluate

# Load all available predictions
predictions = {}
versions = [3, 5, 1, 8]

for v in versions:
    try:
        data = np.load(f"predictions_v{v}.npz")
        predictions[v] = data['gamma_scores']
        print(f"✓ Loaded v{v}")
    except FileNotFoundError:
        print(f"✗ Missing v{v}")

print("\n" + "="*70)
print("WEIGHTED ENSEMBLE EXPERIMENTS")
print("="*70)

# Define ensembles: (attempt_num, weights_dict, description)
ensembles = [
    (25, {3: 0.25, 5: 0.25, 1: 0.25, 8: 0.25}, "Equal weight: 25% each"),
    (26, {3: 0.7, 5: 0.1, 1: 0.1, 8: 0.1}, "v3-heavy: 70% v3, 10% each"),
    (27, {3: 0.6, 8: 0.4}, "Top 2: 60% v3, 40% v8"),
    (28, {3: 0.5, 5: 0.25, 1: 0.25}, "v3+v5+v1: 50/25/25"),
]

results = []
for attempt, weights, desc in ensembles:
    if all(v in predictions for v in weights.keys()):
        # Compute weighted ensemble
        ensemble = np.zeros_like(predictions[list(weights.keys())[0]])
        weight_sum = sum(weights.values())
        
        for v, w in weights.items():
            ensemble += (w / weight_sum) * predictions[v]
        
        # Save and evaluate
        np.savez(f"predictions_v{attempt}.npz", gamma_scores=ensemble)
        metric = evaluate(ensemble, f"v{attempt}: {desc}")
        results.append((attempt, metric, desc))
        print()

if results:
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    results.sort(key=lambda x: x[1])
    for attempt, metric, desc in results:
        print(f"v{attempt}: {metric:.4e}  — {desc}")
