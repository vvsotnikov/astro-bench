#!/bin/bash
echo "Waiting for v16 to complete (seed=456)..."
start_time=$(date +%s)

while [ ! -f predictions_v16.npz ]; do
  elapsed=$(($(date +%s) - start_time))
  echo "[$(date '+%H:%M:%S')] Elapsed: ${elapsed}s - v16 still training..."
  sleep 30
done

echo "✓ V16 COMPLETE at $(date)"
echo ""
echo "=== V16 Results ==="
tail -1 results.tsv | awk -F'\t' '{print "Metric:", $2, "\nDescription:", $4}'
