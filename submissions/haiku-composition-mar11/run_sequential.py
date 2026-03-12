#!/usr/bin/env python3
"""
Sequential experiment runner: one GPU experiment at a time.

This ensures only ONE training process uses the GPU at any moment,
preventing OOM and GPU contention.

Usage:
  python run_sequential.py [--start v14] [--end v26]

Will run v14, v15, v16, ... v26 sequentially, waiting for each to complete
before starting the next.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

SUBMISSION_DIR = Path("submissions/haiku-composition-mar11")
TIMEOUT_SECONDS = 3600  # 1 hour per experiment

# All experiments in order
EXPERIMENTS = [
    "v14",  # v1 + 100 epochs, lr=3e-4
    "v15",  # v1 + class weights
    "v16",  # v1 + label smoothing 0.1
    "v17",  # Exact haiku-mar8 replica
    "v18",  # Deeper CNN (5 blocks)
    "v19",  # Wider CNN
    "v20",  # Vision Transformer 2x2
    "v21",  # MLP flattened
    "v22",  # RandomForest (CPU, can run in parallel)
    "v23",  # Focal loss
    "v24",  # SGD momentum
    "v26",  # Augmentation
]

def get_script_path(version):
    """Convert version (v14) to script path."""
    script_map = {
        "v14": "train_v14_v1_long_training.py",
        "v15": "train_v15_v1_class_weights.py",
        "v16": "train_v16_v1_label_smoothing.py",
        "v17": "train_v17_haiku_mar8_exact.py",
        "v18": "train_v18_deeper_cnn.py",
        "v19": "train_v19_wider_cnn.py",
        "v20": "train_v20_vit_2x2.py",
        "v21": "train_v21_mlp_flattened.py",
        "v22": "train_v22_randomforest_safe.py",
        "v23": "train_v23_focal_loss.py",
        "v24": "train_v24_sgd_momentum.py",
        "v26": "train_v26_augmentation.py",
    }
    return SUBMISSION_DIR / script_map[version]

def is_gpu_experiment(version):
    """Check if experiment uses GPU."""
    return version != "v22"  # Only v22 (RandomForest) is CPU-only

def run_experiment(version):
    """Run a single experiment and wait for completion."""
    script_path = get_script_path(version)
    log_path = SUBMISSION_DIR / f"train_{version}.log"

    if not script_path.exists():
        print(f"ERROR: {script_path} not found")
        return False

    print(f"\n{'='*70}")
    print(f"Starting {version} ({script_path.name})")
    print(f"Log: {log_path}")
    print(f"{'='*70}")

    # Build command
    cmd = ["uv", "run", "python", str(script_path)]

    # Add GPU environment variables for GPU experiments
    env = os.environ.copy()
    if is_gpu_experiment(version):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = "1"

    # Run with timeout
    try:
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )

            print(f"Process {process.pid} started")

            # Wait for completion with timeout
            try:
                return_code = process.wait(timeout=TIMEOUT_SECONDS)
                elapsed = TIMEOUT_SECONDS  # Actual elapsed would be less
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT after {TIMEOUT_SECONDS}s - killing process")
                process.kill()
                process.wait()
                return False

            if return_code == 0:
                print(f"✓ {version} completed successfully")
                return True
            else:
                print(f"✗ {version} failed with return code {return_code}")
                return False

    except Exception as e:
        print(f"ERROR running {version}: {e}")
        return False

def extract_result(version):
    """Extract metric from log file."""
    log_path = SUBMISSION_DIR / f"train_{version}.log"
    if not log_path.exists():
        return None

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('metric:'):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    except:
        pass

    return None

def main():
    # Parse arguments
    start_version = "v14"
    end_version = "v26"

    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(__doc__)
            return
        elif sys.argv[1] == "--start" and len(sys.argv) > 2:
            start_version = sys.argv[2]

    if len(sys.argv) > 3:
        if sys.argv[3] == "--end" and len(sys.argv) > 4:
            end_version = sys.argv[4]

    # Filter experiments
    start_idx = EXPERIMENTS.index(start_version) if start_version in EXPERIMENTS else 0
    end_idx = EXPERIMENTS.index(end_version) if end_version in EXPERIMENTS else len(EXPERIMENTS) - 1

    to_run = EXPERIMENTS[start_idx:end_idx+1]

    print(f"Sequential Runner: {len(to_run)} experiments")
    print(f"GPU: CUDA_VISIBLE_DEVICES=1")
    print(f"Experiments: {to_run}")
    print(f"Total estimated time: {len([v for v in to_run if is_gpu_experiment(v)]) * 35} minutes for GPU")
    print()

    completed = 0
    failed = 0
    results = {}

    # Run experiments sequentially
    for version in to_run:
        success = run_experiment(version)

        if success:
            metric = extract_result(version)
            results[version] = metric
            completed += 1
            print(f"Result: {metric}" if metric else "Result: (not found in log)")
        else:
            failed += 1
            results[version] = "FAILED"

        # Wait a moment before next experiment (cleanup)
        time.sleep(5)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {completed} completed, {failed} failed")
    print(f"{'='*70}")
    for version, result in results.items():
        status = "✓" if result != "FAILED" and result else "✗"
        print(f"{status} {version}: {result}")

if __name__ == "__main__":
    main()
