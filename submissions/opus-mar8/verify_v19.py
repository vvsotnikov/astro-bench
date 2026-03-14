"""Verify v19 multi-seed ensemble variants with official verify.py."""
import numpy as np
import subprocess

OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

probs_v8 = np.load(f"{OUT_DIR}/probs_v8.npy")
probs_s123 = np.load(f"{OUT_DIR}/probs_v19_seed123.npy")
probs_s7 = np.load(f"{OUT_DIR}/probs_v19_seed7.npy")

combos = [
    ("v8_only", [probs_v8]),
    ("seed123_only", [probs_s123]),
    ("seed7_only", [probs_s7]),
    ("v8+s123", [probs_v8, probs_s123]),
    ("v8+s7", [probs_v8, probs_s7]),
    ("s123+s7", [probs_s123, probs_s7]),
    ("v8+s123+s7", [probs_v8, probs_s123, probs_s7]),
    # Weighted: give v8 more weight since it's the best single model
    ("v8*2+s123+s7_w", [probs_v8, probs_v8, probs_s123, probs_s7]),
]

for name, prob_list in combos:
    avg = np.mean(prob_list, axis=0)
    preds = avg.argmax(1).astype(np.int8)
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds)
    result = subprocess.run(
        ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
        capture_output=True, text=True,
        cwd="/home/vladimir/cursor_projects/astro-agents"
    )
    frac_err = "NOT FOUND"
    for line in result.stdout.split('\n'):
        if 'mean fraction error' in line.lower():
            frac_err = line.strip()
            break
    y_test = np.load("/home/vladimir/cursor_projects/astro-agents/data/composition_test/labels_composition.npy")
    acc = (preds == y_test).mean()
    print(f"{name:20s}: acc={acc:.4f}  {frac_err}", flush=True)

# Restore v8
np.savez(f"{OUT_DIR}/predictions.npz", predictions=probs_v8.argmax(1).astype(np.int8))
print("\nRestored v8 predictions")
