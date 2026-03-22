#!/bin/bash
# Monitor phases 2→3→4 and auto-queue

echo "[$(date '+%H:%M')] Starting comprehensive monitor..."

# Phase 2→3
echo "[$(date '+%H:%M')] Waiting for Phase 2 completion (v30-v35)..."
while true; do
  count=0
  for v in v30 v31 v32 v33 v34 v35; do
    [ -f "predictions_${v}.npz" ] && ((count++))
  done
  if [ $count -eq 6 ]; then
    echo "[$(date '+%H:%M')] ✓ Phase 2 complete, queueing Phase 3"
    nohup bash gpu_pipeline_phase3.sh > gpu_phase3.log 2>&1 &
    break
  fi
  echo "[$(date '+%H:%M')] Phase 2: $count/6 complete"
  sleep 30
done

# Phase 3→4
echo "[$(date '+%H:%M')] Waiting for Phase 3 completion (v36-v40)..."
while true; do
  count=0
  for v in v36 v37 v38 v39 v40; do
    [ -f "predictions_${v}.npz" ] && ((count++))
  done
  if [ $count -eq 5 ]; then
    echo "[$(date '+%H:%M')] ✓ Phase 3 complete, queueing Phase 4"
    nohup bash gpu_pipeline_phase4.sh > gpu_phase4.log 2>&1 &
    break
  fi
  echo "[$(date '+%H:%M')] Phase 3: $count/5 complete"
  sleep 30
done

echo "[$(date '+%H:%M')] All phases queued successfully"
