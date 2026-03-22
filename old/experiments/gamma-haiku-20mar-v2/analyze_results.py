"""Analyze all experiment results and identify patterns."""
import numpy as np

# Read results
with open("results.tsv") as f:
    lines = f.readlines()

results = []
for line in lines[1:]:  # Skip header
    parts = line.strip().split("\t")
    if len(parts) >= 4:
        attempt = int(parts[0])
        metric = float(parts[1])
        desc = parts[3]
        results.append((attempt, metric, desc))

results.sort(key=lambda x: x[1])

print("=" * 80)
print("GAMMA CLASSIFICATION RESULTS ANALYSIS")
print("=" * 80)
print()

print("TOP 10 RESULTS:")
print("-" * 80)
for attempt, metric, desc in results[:10]:
    desc_short = desc[:65]
    print(f"{attempt:2d}. {metric:.4e}  — {desc_short}")
print()

# Identify patterns
print("ARCHITECTURE BREAKDOWN:")
print("-" * 80)
mlps = [r for r in results if 'MLP' in r[2] or 'Residual' in r[2]]
cnns = [r for r in results if 'CNN' in r[2] or 'Conv' in r[2]]
ensembles = [r for r in results if 'Ensemble' in r[2]]
rf = [r for r in results if 'Random Forest' in r[2]]

if mlps:
    print(f"MLPs (n={len(mlps)}):    best = {mlps[0][1]:.4e}")
if cnns:
    print(f"CNNs (n={len(cnns)}):    best = {cnns[0][1]:.4e}")
if ensembles:
    print(f"Ensembles (n={len(ensembles)}): best = {ensembles[0][1]:.4e}")
if rf:
    print(f"RF (n={len(rf)}):      {rf[0][1]:.4e}")
print()

print("OVERALL STATS:")
metrics = [r[1] for r in results]
print(f"  Total experiments: {len(results)}")
print(f"  Mean: {np.mean(metrics):.4e}")
print(f"  Median: {np.median(metrics):.4e}")
print(f"  Best: {np.min(metrics):.4e}")
print(f"  Worst: {np.max(metrics):.4e}")
print(f"  Improvement vs baseline (1e-2): {(1e-2 / np.min(metrics)):.1f}×")
