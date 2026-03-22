#!/bin/bash
set -e

echo "=== RAPID ENSEMBLE PHASE ==="
SEEDS=(456 789 999 1111)
VERSIONS=(16 17 18 19)

for i in "${!SEEDS[@]}"; do
  seed=${SEEDS[$i]}
  ver=${VERSIONS[$i]}
  echo "Starting v$ver (seed=$seed)..."
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v${ver}.py > submissions/v1/v${ver}.log 2>&1
  echo "v$ver done"
  sleep 2
done

echo "All seeds trained. Creating ensemble..."
python -c "
import numpy as np
from verify import evaluate
preds = []
weights = []
for v in [3, 16, 17, 18, 19]:
    try:
        p = np.load(f'predictions_v{v}.npz')['gamma_scores']
        preds.append(p)
        weights.append(1.0 / len([x for x in [3,16,17,18,19]]))
    except:
        print(f'v{v} not found')

if preds:
    ensemble = sum(w * p for w, p in zip(weights, preds))
    np.savez('predictions_v20_ensemble.npz', gamma_scores=ensemble)
    evaluate(ensemble, 'v20: Ensemble v3+v16+v17+v18+v19 (5-seed average)')
"
