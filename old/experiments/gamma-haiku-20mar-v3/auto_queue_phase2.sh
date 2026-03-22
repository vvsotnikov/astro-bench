#!/bin/bash
# Auto-queue Phase 2 when Phase 1 CPU models finish

echo "Monitoring Phase 1 CPU models (v15, v20, v21)..."
timeout 3600 bash << 'INNER'
while true; do
  ready=0
  for model in v15 v20 v21; do
    if [ -f "predictions_${model}.npz" ]; then
      ready=$((ready + 1))
    fi
  done
  
  if [ $ready -eq 3 ]; then
    echo ""
    echo "=== Phase 1 CPU Complete ==="
    echo "Queuing Phase 2: v28-v32"
    break
  fi
  
  sleep 30
done
INNER

# Phase 2: Sequential on CPU (no GPU conflict)
for model in v28 v29 v30 v31 v32; do
  if [ -f "train_${model}.py" ]; then
    echo ""
    echo "Starting Phase 2: $model"
    timeout 2700 uv run python "train_${model}.py" > "${model}.log" 2>&1
    if [ -f "predictions_${model}.npz" ]; then
      echo "✓ $model complete"
    fi
  fi
done

echo ""
echo "=== Phase 2 Complete ==="
echo "Queuing Phase 3: v33-v35"

# Phase 3: Channel ablation and ensemble
for model in v33 v34 v35; do
  if [ -f "train_${model}.py" ]; then
    echo ""
    echo "Starting Phase 3: $model"
    timeout 2700 uv run python "train_${model}.py" > "${model}.log" 2>&1
    if [ -f "predictions_${model}.npz" ]; then
      echo "✓ $model complete"
    fi
  fi
done

echo ""
echo "=== Exploration Phases 1-3 Complete ==="
echo "Results in results.tsv"
