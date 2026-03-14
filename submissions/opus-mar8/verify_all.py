"""Verify fraction error for all saved probability files."""
import numpy as np
import subprocess
import os

OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"

y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)

prob_files = [
    ("v5", f"{OUT_DIR}/probs_v5.npy"),
    ("v8", f"{OUT_DIR}/probs_v8.npy"),
    ("v9_ep10", f"{OUT_DIR}/probs_v9_ep10.npy"),
    ("v15_hgb", f"{OUT_DIR}/probs_v15.npy"),
    ("v18_tta", f"{OUT_DIR}/probs_v18.npy"),
    ("hgb", f"{OUT_DIR}/probs_hgb.npy"),
]

for name, path in prob_files:
    if not os.path.exists(path):
        print(f"{name}: FILE NOT FOUND")
        continue
    probs = np.load(path)
    preds = probs.argmax(1)
    acc = (preds == y_test).mean()

    # Save and verify
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))
    result = subprocess.run(
        ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
        capture_output=True, text=True, cwd="/home/vladimir/cursor_projects/astro-agents"
    )
    # Extract fraction error from output
    for line in result.stdout.split('\n'):
        if 'mean fraction error' in line.lower():
            frac_err = line.strip()
            break
    else:
        frac_err = "NOT FOUND"

    print(f"{name:15s}: acc={acc:.4f}  {frac_err}")

# Restore v8
probs_v8 = np.load(f"{OUT_DIR}/probs_v8.npy")
np.savez(f"{OUT_DIR}/predictions.npz", predictions=probs_v8.argmax(1).astype(np.int8))
print("\nRestored v8 predictions")
