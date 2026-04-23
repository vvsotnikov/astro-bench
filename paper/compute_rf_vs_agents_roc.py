"""Compare RF baseline vs best agent at multiple gamma efficiencies."""
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/gamma')
from verify import _survival_at_efficiency, _load_test_labels
import numpy as np

labels = _load_test_labels()
n_g = (labels == 0).sum()
n_h = (labels == 1).sum()
print(f"Test: {len(labels)} events, {n_g} gammas, {n_h} hadrons\n")

rf_scores = np.load('/home/vladimir/cursor_projects/astro-agents/gamma/baseline/predictions_rf.npz')['predictions']

# Best gamma scores per agent
agents = {
    'Opus 4.6':   ('/home/vladimir/cursor_projects/astro-bench-experiments/gamma-opus-23mar/predictions_v5.npz',          'ensemble_geo'),
    'GPT-5.4':    ('/home/vladimir/cursor_projects/astro-bench-experiments/gamma-gpt-25mar/predictions_v31.npz',          None),
    'Sonnet 4.6': ('/home/vladimir/cursor_projects/astro-bench-experiments/gamma-sonnet-22mar-v2/predictions_v28.npz',    None),
    'Kimi K2.5':  ('/home/vladimir/cursor_projects/astro-bench-experiments/gamma-kimi-26mar/predictions_v27.npz',         None),
    'Qwen 3.6':   ('/home/vladimir/cursor_projects/astro-bench-experiments/gamma-qwen-6apr/predictions_v14.npz',          None),
}

efficiencies = [0.30, 0.50, 0.70, 0.75, 0.90]

# RF
print(f"{'efficiency':>12s}  {'RF':>12s}  " + "  ".join(f"{n:>12s}" for n in agents.keys()))
print("-" * (14 + 14 + 14 * len(agents)))

# Load all agent scores first
agent_scores = {}
for name, (path, key) in agents.items():
    try:
        d = np.load(path)
        if key is None:
            keys = list(d.keys())
            # Pick the key likely to be the score (avoid 'predictions' if hard labels)
            for k in keys:
                arr = d[k]
                if arr.dtype in (np.float32, np.float64) and arr.shape == labels.shape:
                    if arr.min() >= 0 and arr.max() <= 1:
                        key = k
                        break
            if key is None:
                # fallback to first
                key = keys[0]
        agent_scores[name] = (d[key], path, key)
    except FileNotFoundError:
        # try to find any predictions_* file in the dir
        from pathlib import Path
        d = Path(path).parent
        files = sorted(d.glob('predictions_v*.npz'))
        if files:
            df = np.load(files[-1])
            print(f"  WARN: {path} not found, using {files[-1]}")
            agent_scores[name] = (df[list(df.keys())[0]], str(files[-1]), list(df.keys())[0])
        else:
            agent_scores[name] = None

def surv(scores, eff):
    g_mask = labels == 0
    h_mask = labels == 1
    return _survival_at_efficiency(scores[g_mask], scores[h_mask], eff)['hadron_survival']

for eff in efficiencies:
    rf_s = surv(rf_scores, eff)
    row = [f"{eff:>12.2f}", f"{rf_s:>12.3e}"]
    for name in agents.keys():
        if agent_scores.get(name) is None:
            row.append(f"{'--':>12s}")
            continue
        s = surv(agent_scores[name][0], eff)
        row.append(f"{s:>12.3e}")
    print("  ".join(row))

print()
for name, info in agent_scores.items():
    if info:
        print(f"  {name}: {info[1]} key={info[2]}")
