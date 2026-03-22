#!/bin/bash
# CPU pipeline for v46-v50 - can run some in parallel since they're CPU-based

echo "[$(date '+%H:%M')] Phase 5 CPU pipeline starting..."

# v46, v47 are sklearn models (pure CPU) - can run somewhat in parallel
export CUDA_VISIBLE_DEVICES=1
timeout 1200 uv run python train_v46.py > v46.log 2>&1 &
pid46=$!
timeout 1200 uv run python train_v47.py > v47.log 2>&1 &
pid47=$!
wait $pid46 $pid47

# v48, v49, v50 are GPU/ensemble models - run sequentially 
for model in v48 v49 v50; do
  echo "[$(date '+%H:%M')] Starting $model..."
  if [ "$model" = "v49" ]; then
    timeout 900 uv run python train_${model}.py > ${model}.log 2>&1
  else
    timeout 4800 uv run python train_${model}.py > ${model}.log 2>&1
  fi
  
  if [ -f "predictions_${model}.npz" ]; then
    echo "[$(date '+%H:%M')] ✓ $model COMPLETE"
  else
    echo "[$(date '+%H:%M')] ✗ $model FAILED"
  fi
  sleep 5
done

echo "[$(date '+%H:%M')] Phase 5 pipeline complete"
