#!/bin/bash

echo "Monitoring all models..." 
start_time=$(date +%s)

while true; do
  elapsed=$(($(date +%s) - start_time))
  
  ready=0
  for model in v15 v20 v21 v23 v24 v25; do
    if [ -f "predictions_${model}.npz" ]; then
      result=$(tail -1 results.tsv | awk -F'\t' '{print $2}')
      echo "[$(date '+%H:%M:%S')] ✓ $model COMPLETE - $result"
      ready=$((ready + 1))
    fi
  done
  
  # Stop after 2 hours or when 6 models complete
  if [ $ready -ge 6 ] || [ $elapsed -gt 7200 ]; then
    break
  fi
  
  sleep 30
done
