#!/bin/bash
# Monitor phase 2 completion and queue phase 3

echo "[$(date '+%H:%M')] Starting phase 2→3 monitor..."

# Wait for phase 2 to complete (all 6 models have predictions)
while true; do
  count=0
  for v in v30 v31 v32 v33 v34 v35; do
    [ -f "predictions_${v}.npz" ] && ((count++))
  done
  
  if [ $count -eq 6 ]; then
    echo "[$(date '+%H:%M')] ✓ Phase 2 complete (6/6 models)"
    echo "[$(date '+%H:%M')] Launching Phase 3..."
    nohup bash gpu_pipeline_phase3.sh > gpu_phase3.log 2>&1 &
    break
  else
    echo "[$(date '+%H:%M')] Phase 2: $count/6 models complete"
    sleep 30
  fi
done

echo "[$(date '+%H:%M')] Phase 3 queued, monitor process exiting"
