# KASCADE Cosmic Ray Classification Challenge

## The Task

Build a model that classifies cosmic ray primary particles into 5 mass groups based on extensive air shower measurements from the KASCADE experiment:

| Class | Particle | Index |
|-------|----------|-------|
| 0 | Proton (H) | 14 |
| 1 | Helium (He) | 402 |
| 2 | Carbon (C) | 1206 |
| 3 | Silicon (Si) | 2814 |
| 4 | Iron (Fe) | 5626 |

Your goal: maximize classification accuracy on a held-out test set.

## The Data

All data is in `./data/` after running `python download_data.py`.

### Matrix data (primary dataset)

16x16 detector grid images from KASCADE's array of 252 detector stations. Each event is a 16x16x3 array:
- Channel 0: arrival time differences
- Channel 1: electron/photon densities
- Channel 2: muon densities

Files (NPZ format):
- `{mode}_matrices.npz` — key: `matrices`, shape: `(N, 16, 16, 3)`
- `{mode}_features.npz` — key: `features`, shape: `(N, 10)`
- `{mode}_true_features.npz` — key: `true_features`, shape: `(N, 10)`

Where `mode` is `qgs_spectra` or `epos_spectra` (two hadronic interaction models used in simulation).

**Reconstructed features** (what a real detector would measure):
```
features[:, 0]  = particle type (reconstructed)
features[:, 1]  = E       — log10(energy/eV), reconstructed
features[:, 2]  = Xc      — shower core X position
features[:, 3]  = Yc      — shower core Y position
features[:, 4]  = core_dist — distance from array center
features[:, 5]  = Ze      — zenith angle (degrees)
features[:, 6]  = Az      — azimuth angle (degrees)
features[:, 7]  = Ne      — log10(electron number)
features[:, 8]  = Nmu     — log10(muon number)
features[:, 9]  = Age     — shower age parameter
```

**True (simulated) features** (ground truth from Monte Carlo):
```
true_features[:, 0] = E         — log10(true energy/eV)
true_features[:, 1] = part_type — particle ID (1=gamma, 14=H, 402=He, 1206=C, 2814=Si, 5626=Fe)
true_features[:, 2] = Xc        — true core X
true_features[:, 3] = Yc        — true core Y
true_features[:, 4] = Ze        — true zenith
true_features[:, 5] = Az        — true azimuth
true_features[:, 6] = Ne        — true electron number
true_features[:, 7] = Np        — true photon number
true_features[:, 8] = Nmu       — true muon number
true_features[:, 9] = Nh        — true hadron number
```

### Tabular data (alternative representation)

CSV files with array-level observables: `data_array_{epos-LHC,qgs-4,sibyll-23c}.csv`

Columns: `E, Xc, Yc, Ze, Az, Ne, Nmu, Age, p_idx, hadron`

Where `p_idx` is the particle class (1-5 mapping to proton through iron).

### Train/test split

`data/test_split.npz` contains `train_indices` and `test_indices` into the concatenated matrix dataset (qgs_spectra + epos_spectra). **This split is fixed (seed=2026). Do not modify it.**

## Quality Cuts (from published analyses)

Previous analyses applied these cuts. You may choose to apply them or not:
- Ze < 18-20 degrees (zenith angle)
- Ne > 4.8 (log10 electron number)
- Nmu > 3.6 (log10 muon number)
- 0.2 < Age < 1.48 (shower age)

## What to Submit

Your submission directory must contain:
1. `predictions.npz` — with key `predictions`: integer array of class labels (0-4) for each test sample (in test_indices order)
2. `README.md` — document your approach: what you tried, what failed, what worked, architecture details, training curves, anything interesting you found
3. All source code needed to reproduce the result

### If your solution was built by an AI agent, also include:
- `metadata.yaml` with: agent model, prompt/config used, total cost, token count, wall-clock time, number of iterations
- Full reasoning trace if available

## Evaluation

```bash
python verify.py submissions/your_submission/predictions.npz
```

Primary metric: **overall accuracy**. We also report per-class F1, confusion matrix, and energy-binned accuracy.

## Baselines

The published human baselines (Kuznetsov, Petrov, Plokhikh, Sotnikov — JCAP 2024, JINST 2024) represent years of manual iteration by domain experts. The models used attention-based MLPs on digitized 16x16 detector matrices with carefully designed quality cuts and preprocessing.

## Physics Background

When a high-energy cosmic ray hits Earth's atmosphere, it creates a cascade of secondary particles called an extensive air shower. KASCADE measured these showers with a 200x200m array of 252 detector stations in Karlsruhe, Germany, over ~25 years.

The classification task is hard because:
- Air showers are inherently stochastic — the same primary particle can produce very different showers
- The mapping from observables to primary particle is many-to-one and noisy
- Different mass groups overlap significantly in observable space, especially at lower energies
- The hadronic interaction models (QGSJet, EPOS, SIBYLL) used in simulation don't perfectly match real data

Light particles (protons) produce deeper, more variable showers with fewer muons. Heavy particles (iron) produce shallower, more regular showers with more muons. The electron-to-muon ratio (Ne/Nmu) is the strongest discriminant, but it's far from sufficient alone.
