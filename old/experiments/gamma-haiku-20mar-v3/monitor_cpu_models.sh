#!/bin/bash
echo "Monitoring v15, v20, v21, v22..."
checked=0
while [ $checked -lt 100 ]; do
  ready=0
  for model in v15 v20 v21 v22; do
    if [ -f "predictions_${model}.npz" ]; then
      echo "[$(date '+%H:%M:%S')] ✓ $model COMPLETE"
      ready=$((ready + 1))
    fi
  done
  
  if [ $ready -eq 4 ]; then
    echo "All 4 CPU models complete!"
    echo "=== RESULTS ==="
    tail -4 results.tsv | awk -F'\t' '{print $NF}'
    break
  fi
  
  echo "[$(date '+%H:%M:%S')] Ready: $ready/4"
  sleep 10
  checked=$((checked + 1))
done
