"""Monitor active experiments and extract results.

Usage: python monitor_experiments.py [--watch]
       --watch: continuous monitoring every 30s

This script:
1. Checks for completed log files
2. Extracts metric and description
3. Updates results.tsv
4. Reports progress
"""

import os
import sys
import time
import re
from pathlib import Path

SUBMISSION_DIR = Path("submissions/haiku-composition-mar11")
RESULTS_FILE = SUBMISSION_DIR / "results.tsv"

# Map of experiment version to log file
EXPERIMENTS = {
    "v14": "train_v14.log",
    "v15": "train_v15.log",
    "v16": "train_v16.log",
    "v17": "train_v17.log",
    "v18": "train_v18.log",
    "v19": "train_v19.log",
    "v20": "train_v20.log",
    "v21": "train_v21.log",
    "v22": "train_v22.log",
    "v23": "train_v23.log",
    "v24": "train_v24.log",
}

def get_metric_from_log(log_file):
    """Extract metric and description from log file."""
    if not log_file.exists():
        return None, None

    with open(log_file, 'r') as f:
        lines = f.readlines()

    metric = None
    description = None

    for line in lines:
        if line.startswith('metric:'):
            match = re.search(r'metric:\s*([\d.e\-]+)', line)
            if match:
                metric = float(match.group(1))
        elif line.startswith('description:'):
            description = line.replace('description:', '').strip()

    return metric, description

def update_results_file(version, metric, description):
    """Update results.tsv with new result."""
    if not RESULTS_FILE.exists():
        print(f"Creating {RESULTS_FILE}")
        RESULTS_FILE.write_text("experiment\tmetric\tstatus\tdescription\n")

    content = RESULTS_FILE.read_text()
    lines = content.strip().split('\n')

    # Check if version already in results
    for i, line in enumerate(lines):
        if line.startswith(f"{version}\t"):
            lines[i] = f"{version}\t{metric:.4f}\tkeep\t{description}"
            RESULTS_FILE.write_text('\n'.join(lines) + '\n')
            return True

    # Append new result
    lines.append(f"{version}\t{metric:.4f}\tkeep\t{description}")
    RESULTS_FILE.write_text('\n'.join(lines) + '\n')
    return True

def monitor_once():
    """Check all experiments once."""
    print(f"[{time.strftime('%H:%M:%S')}] Checking experiments...\n")

    completed = 0
    for version, log_file in EXPERIMENTS.items():
        log_path = SUBMISSION_DIR / log_file
        metric, description = get_metric_from_log(log_path)

        if metric is not None:
            print(f"✓ {version}: {metric:.4f} — {description}")
            update_results_file(version, metric, description)
            completed += 1
        elif log_path.exists() and log_path.stat().st_size > 1000:
            print(f"~ {version}: Running (log size: {log_path.stat().st_size} bytes)")
        elif log_path.exists():
            print(f"~ {version}: Initializing...")
        else:
            print(f"- {version}: Not started")

    print(f"\nCompleted: {completed}/{len(EXPERIMENTS)}")
    return completed == len(EXPERIMENTS)

def watch_mode():
    """Continuously monitor."""
    print("Watching experiments (Ctrl+C to stop)\n")
    try:
        while True:
            all_done = monitor_once()
            if all_done:
                print("\n✓ All experiments complete!")
                break
            time.sleep(30)
            print()
    except KeyboardInterrupt:
        print("\n\nStopped.")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        watch_mode()
    else:
        monitor_once()
        print("\nUse --watch to continuously monitor")

if __name__ == "__main__":
    main()
