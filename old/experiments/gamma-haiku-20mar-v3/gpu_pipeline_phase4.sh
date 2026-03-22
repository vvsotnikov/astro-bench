#!/bin/bash
# GPU pipeline for v41-v45 - run sequentially with proper CUDA setup

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

for model in v41 v42 v43 v44 v45; do
  echo "[$(date '+%H:%M')] Starting $model..."
  
  if [ "$model" = "v44" ]; then
    # v44 is an ensemble that waits for v30,v31,v32
    timeout 600 uv run python train_${model}.py > ${model}.log 2>&1
  elif [ "$model" = "v45" ]; then
    # v45 uses test-time dropout (10 passes)
    timeout 6000 uv run python train_${model}.py > ${model}.log 2>&1
  else
    timeout 4800 uv run python train_${model}.py > ${model}.log 2>&1
  fi
  
  if [ -f "predictions_${model}.npz" ]; then
    echo "[$(date '+%H:%M')] ✓ $model COMPLETE"
    grep "${model}:" results.tsv | tail -1
  else
    echo "[$(date '+%H:%M')] ✗ $model FAILED (no predictions)"
  fi
  
  sleep 5
done

echo "[$(date '+%H:%M')] Phase 4 GPU pipeline complete"
