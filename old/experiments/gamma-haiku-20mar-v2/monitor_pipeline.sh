#!/bin/bash
echo "=== Gamma Classification Pipeline Monitor ==="
echo "Started: $(date)"
echo ""

while true; do
  echo "[$(date '+%H:%M:%S')] Status:"
  
  # Check phase 1 (seeds)
  for v in 16 17 18 19; do
    if [ -f predictions_v${v}.npz ]; then
      tail_line=$(tail -1 results.tsv | awk -F'\t' '{print $2}')
      echo "  v${v}: ✓ COMPLETE (${tail_line})"
    elif pgrep -f "train_v${v}\.py" > /dev/null; then
      echo "  v${v}: ⏳ training..."
    else
      echo "  v${v}: ⌛ queued"
    fi
  done
  
  # Check ensemble results
  if [ -f predictions_v20_ensemble.npz ]; then
    echo "  v20 (5-seed ensemble): ✓ COMPLETE"
  fi
  
  # Check phase 2
  for v in 21 22; do
    if [ -f predictions_v${v}.npz ]; then
      echo "  v${v}: ✓ COMPLETE"
    elif pgrep -f "train_v${v}\.py" > /dev/null; then
      echo "  v${v}: ⏳ training..."
    fi
  done
  
  # Show best so far
  best=$(tail -1 results.tsv | awk -F'\t' '{print $2}')
  best_v=$(tail -1 results.tsv | awk -F'\t' '{print $4}')
  echo ""
  echo "  Best: ${best} (${best_v})"
  
  sleep 30
done
