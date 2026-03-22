#!/bin/bash
# Monitor all 5 phases and report final results

echo "[$(date '+%H:%M')] === MASTER MONITOR STARTING ==="

# Wait for each phase and auto-queue next
phases=("Phase 2" "Phase 3" "Phase 4" "Phase 5")
models_per_phase=(6 5 5 5)
model_ranges=("v30-v35" "v36-v40" "v41-v45" "v46-v50")

for phase_idx in {0..3}; do
  phase_name="${phases[$phase_idx]}"
  num_models="${models_per_phase[$phase_idx]}"
  model_range="${model_ranges[$phase_idx]}"
  
  echo "[$(date '+%H:%M')] Waiting for $phase_name ($model_range)..."
  
  while true; do
    completed=0
    # Count prediction files for this phase
    case $phase_idx in
      0) for v in v30 v31 v32 v33 v34 v35; do [ -f "predictions_${v}.npz" ] && ((completed++)); done ;;
      1) for v in v36 v37 v38 v39 v40; do [ -f "predictions_${v}.npz" ] && ((completed++)); done ;;
      2) for v in v41 v42 v43 v44 v45; do [ -f "predictions_${v}.npz" ] && ((completed++)); done ;;
      3) for v in v46 v47 v48 v49 v50; do [ -f "predictions_${v}.npz" ] && ((completed++)); done ;;
    esac
    
    if [ $completed -eq $num_models ]; then
      echo "[$(date '+%H:%M')] ✓ $phase_name COMPLETE ($completed/$num_models)"
      
      # Queue next phase if available
      if [ $phase_idx -lt 3 ]; then
        next_pipeline=""
        case $phase_idx in
          0) next_pipeline="gpu_pipeline_phase3.sh" ;;
          1) next_pipeline="gpu_pipeline_phase4.sh" ;;
          2) next_pipeline="cpu_pipeline_phase5.sh" ;;
        esac
        if [ ! -z "$next_pipeline" ] && [ -f "$next_pipeline" ]; then
          echo "[$(date '+%H:%M')] Queueing ${phases[$((phase_idx+1))]}..."
          nohup bash $next_pipeline > gpu_phase$((phase_idx+2)).log 2>&1 &
        fi
      fi
      break
    else
      echo "[$(date '+%H:%M')] $phase_name: $completed/$num_models complete"
      sleep 45
    fi
  done
done

echo "[$(date '+%H:%M')] === ALL PHASES COMPLETE ==="
echo "[$(date '+%H:%M')] Final Results:"
tail -10 results.tsv
echo "[$(date '+%H:%M')] Best result: $(tail -10 results.tsv | head -1)"
