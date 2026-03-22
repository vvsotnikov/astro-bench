#!/bin/bash
# Background experiment monitor - runs until all 50 attempts complete

echo "[$(date '+%H:%M:%S')] Experiment monitor started"

# Monitor Phases 3-5 until complete
phases=(
  "Phase 3:v36:v37:v38:v39:v40"
  "Phase 4:v41:v42:v43:v44:v45"
  "Phase 5:v46:v47:v48:v49:v50"
)

for phase_spec in "${phases[@]}"; do
  IFS=':' read -r phase_name models <<< "$phase_spec"
  models=(${models})
  
  echo "[$(date '+%H:%M:%S')] Waiting for $phase_name (${models[@]})"
  
  # Wait for all models in this phase to complete
  while true; do
    completed=0
    for model in "${models[@]}"; do
      [ -f "predictions_${model}.npz" ] && ((completed++))
    done
    
    if [ $completed -eq ${#models[@]} ]; then
      echo "[$(date '+%H:%M:%S')] ✓ $phase_name complete ($completed/${#models[@]})"
      break
    else
      echo "[$(date '+%H:%M:%S')] $phase_name: $completed/${#models[@]} complete"
      sleep 60
    fi
  done
done

# Final summary
echo "[$(date '+%H:%M:%S')] === EXPERIMENT COMPLETE ==="
echo "Total attempts: $(( $(wc -l < results.tsv) - 1 ))"
echo "Best result: $(tail -5 results.tsv | sort -t$'\t' -k2 -g | head -1 | cut -f2,4)"
echo "[$(date '+%H:%M:%S')] Monitor exiting"
