"""Bar chart: best-single vs naive ensemble vs oracle on composition.

Visualizes the oracle headroom — the most novel quantitative finding. Shows
that all five agents converge to a narrow band around 0.105, naive majority
voting does not improve, and an oracle selector across three best agents
reaches 0.0958 — a ~20x training-recipe-noise-floor headroom that simple
aggregation fails to extract.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent

# Per-agent best fraction error from Table 2
AGENTS = [
    ("Kimi K2.5",     0.1063, "agent"),
    ("Qwen 3.6-Plus", 0.1060, "agent"),
    ("Sonnet 4.6",    0.1054, "agent"),
    ("GPT-5.4",       0.1052, "agent"),
    ("Opus 4.6",      0.1051, "agent"),
]

# Realized aggregators (5-seed mean from stacking_experiment.py, 50/50 stratified split)
# and the unrealizable oracle upper bound.
ENSEMBLE = [
    ("Majority vote",                                  0.10509, "ensemble"),
    ("Score average",                                  0.10514, "ensemble"),
    ("Calibrated soft-vote",                           0.10516, "ensemble"),
    ("Stacker (LR, balanced)",                         0.10542, "ensemble"),
    ("Stacker (LR)",                                   0.10567, "ensemble"),
    ("Stacker (MLP)",                                  0.10696, "ensemble"),
    ("MA-MLP (CE+batch-frac, $\\alpha$=100)",         0.10643, "ensemble"),
    ("MA-MLP (verifier-style)",                       0.10566, "ensemble"),
    ("Oracle selector",                                0.09597, "oracle"),
]

NOISE_FLOOR_SIGMA = 4.0e-4  # single-recipe stochasticity, Table 1 / sec:comp_noise_floor
BASELINE = 0.107             # RF reference, Table 2

ALL = AGENTS + ENSEMBLE
labels = [a[0] for a in ALL]
values = [a[1] for a in ALL]
kinds  = [a[2] for a in ALL]

color_for = {"agent": "#4C72B0", "ensemble": "#DD8452", "oracle": "#55A868"}
colors = [color_for[k] for k in kinds]

fig, ax = plt.subplots(figsize=(7.0, 4.6))

ypos = np.arange(len(labels))
ax.barh(ypos, values, color=colors, edgecolor="black", linewidth=0.6, height=0.65)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()  # best on top, oracle at bottom
ax.set_xlabel("Mean fraction error (lower is better)", fontsize=10)

# Reference: noise floor band around best single agent (Opus)
opus_best = 0.1051
ax.axvspan(opus_best - NOISE_FLOOR_SIGMA, opus_best + NOISE_FLOOR_SIGMA,
           color="gray", alpha=0.18, zorder=0,
           label=fr"single-recipe noise floor ($\sigma \approx 4\!\times\!10^{{-4}}$)")

# Reference: published RF baseline (off-scale to the right, drop annotation)
# Avoid clutter; baseline is not in the same regime.

# Value labels at end of each bar
for i, v in enumerate(values):
    ax.text(v + 0.0003, i, f"{v:.4f}", va="center", fontsize=8.5)

# Headroom annotation: span between best realized aggregator and oracle
y_mv = labels.index("Majority vote")
y_oracle = labels.index("Oracle selector")
y_span = (y_mv + y_oracle) / 2.0

ax.annotate("",
            xy=(0.0960, y_span),
            xytext=(0.1051, y_span),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.2,
                            shrinkA=0, shrinkB=0))
ax.text(0.10055, y_span - 0.20,
        r"unrecovered oracle headroom: $\Delta \approx 9 \times 10^{-3}$  ($\approx 20\,\sigma_{\rm noise}$)",
        ha="center", va="bottom", fontsize=8.5, color="black",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

ax.set_xlim(0.094, 0.108)
ax.legend(loc="lower right", fontsize=8.5, frameon=True)
ax.grid(axis="x", alpha=0.3, linestyle=":")
ax.set_axisbelow(True)

plt.tight_layout()
out = HERE / "fig_oracle_headroom.pdf"
plt.savefig(out)
print(f"Saved {out}")
