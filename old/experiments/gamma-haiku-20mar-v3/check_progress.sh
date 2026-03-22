#!/bin/bash
# Quick status check for the autonomous experiment run

echo "=== AUTONOMOUS EXPERIMENT STATUS ==="
echo "Timestamp: $(date)"
echo

echo "Completed Attempts (from results.tsv):"
wc -l results.tsv | awk '{print "  Lines: " $1 " (attempt 0 = baseline, so " $1-2 " experiments + 1 header)"}'
echo "  Best so far: $(tail -1 results.tsv | awk -F'\t' '{print $2 " - " $4}')"
echo

echo "Prediction Files Generated:"
ls predictions_v*.npz 2>/dev/null | wc -l
echo "  Models: v1-v9 (baseline), v30-v50 (in progress)"
echo

echo "Active Processes:"
ps aux | grep -E "gpu_pipeline|final_monitor|train_v[34]" | grep -v grep | wc -l
echo "  Pipelines running"
echo

echo "Next Milestones:"
echo "  [Phase 2] v30-v35 (GPU): Dec by $(ls predictions_v3*.npz 2>/dev/null | wc -l)/6 complete"
echo "  [Phase 3] v36-v40 (GPU): Queued after Phase 2"
echo "  [Phase 4] v41-v45 (GPU): Queued after Phase 3"
echo "  [Phase 5] v46-v50 (CPU): Queued after Phase 4"
echo "  [TOTAL] 50 attempts budgeted"
echo

echo "Estimated remaining time:"
echo "  Each phase: 60-90 min (varies by model complexity)"
echo "  Total: ~5-6 hours from 00:54"
