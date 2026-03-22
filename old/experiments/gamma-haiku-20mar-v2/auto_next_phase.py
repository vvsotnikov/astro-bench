"""Automatically transition to Phase 4 based on Phase 1-3 results."""
import subprocess
import time
from pathlib import Path

def wait_for_phase3_completion():
    """Wait for v23 and v24 to complete."""
    print("Waiting for Phase 3 (v23-v24) completion...")
    while True:
        v23_done = Path("predictions_v23.npz").exists()
        v24_done = Path("predictions_v24.npz").exists()
        
        if v23_done and v24_done:
            print("✓ Phase 3 complete! Both v23 and v24 done.")
            break
        
        time.sleep(30)

def analyze_ensemble_performance():
    """Analyze what worked best to inform Phase 4."""
    with open("results.tsv") as f:
        lines = f.readlines()
    
    results = {}
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            attempt = int(parts[0])
            metric = float(parts[1])
            desc = parts[3]
            results[attempt] = (metric, desc)
    
    # Find Phase 3 results
    phase3_results = {k: v for k, v in results.items() if k >= 23}
    
    print("\n" + "=" * 70)
    print("PHASE 3 RESULTS ANALYSIS")
    print("=" * 70)
    for attempt in sorted(phase3_results.keys()):
        metric, desc = phase3_results[attempt]
        print(f"v{attempt}: {metric:.4e} — {desc}")
    
    best_v3_variant = min(phase3_results.items(), key=lambda x: x[1][0])
    best_metric = best_v3_variant[1][0]
    
    print(f"\nBest variant: v{best_v3_variant[0]} @ {best_metric:.4e}")
    
    # Check if seeds helped
    v20_in_results = any(k == 20 for k in results.keys())
    if v20_in_results:
        v20_metric, _ = results[20]
        v3_metric = results[5][0]
        improvement = (v3_metric - v20_metric) / v3_metric * 100
        print(f"Seed ensemble improvement: {improvement:+.1f}%")
    
    return results, best_metric

if __name__ == "__main__":
    wait_for_phase3_completion()
    results, best_metric = analyze_ensemble_performance()
    
    print("\n" + "=" * 70)
    print("READY FOR PHASE 4 (Weighted Ensembles & Fine-tuning)")
    print("=" * 70)
    print(f"Target to beat: {best_metric:.4e}")
    print("Next: Generate weighted ensemble predictions and test combinations")
