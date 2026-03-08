# KASCADE Cosmic Ray Classification Challenge

## Two Tasks

### Task 1: Mass Composition (5-class classification)

Classify cosmic ray primary particles into 5 mass groups:

| Class | Particle | Raw ID |
|-------|----------|--------|
| 0 | Proton (H) | 14 |
| 1 | Helium (He) | 402 |
| 2 | Carbon (C) | 1206 |
| 3 | Silicon (Si) | 2814 |
| 4 | Iron (Fe) | 5626 |

Gamma rays (ID=1) are excluded from this task.

**Key metric: overall accuracy.**

### Task 2: Gamma/Hadron Separation (binary classification)

Distinguish gamma-ray primaries from hadronic cosmic rays (all other particles).

| Class | Type |
|-------|------|
| 0 | Gamma |
| 1 | Hadron (everything else) |

In real data, the gamma-to-hadron ratio is ~1:1,000,000. The simulation dataset has balanced classes, but evaluation must account for extreme class imbalance in deployment.

**Key metric: hadronic survival rate at 99% gamma efficiency** (i.e., what fraction of hadrons survive your cut while keeping 99% of gammas). Lower is better. Published baseline: 10⁻⁶ to 3×10⁻⁵ (Petrov et al., Chinese Physics C 2023).

## The Data

All data is in `./data/` after running `python download_data.py`.

### Standard input format

Each task has its own pre-split dataset. All files are `.npy` format and support memory-mapping for zero-copy access:

```python
X = np.load('data/composition_train/matrices.npy', mmap_mode='r')  # instant, zero RAM
```

**Task 1 — Mass composition:** `data/composition_train/` and `data/composition_test/`
- Sources: QGSJet-II + EPOS-LHC simulations (~7M hadron events)

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `matrices.npy` | `(N, 16, 16, 2)` | float16 | Detector grid images |
| `features.npy` | `(N, 5)` | float16 | Reconstructed scalar features |
| `labels_composition.npy` | `(N,)` | int8 | Mass class (0-4) |

**Task 2 — Gamma/hadron:** `data/gamma_train/` and `data/gamma_test/`
- Sources: QGSJet-II + EPOS-LHC + SIBYLL gamma+proton simulations (~1.9M events)

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `matrices.npy` | `(N, 16, 16, 2)` | float16 | Detector grid images |
| `features.npy` | `(N, 5)` | float16 | Reconstructed scalar features |
| `labels_gamma.npy` | `(N,)` | int8 | 0=gamma, 1=hadron |

**Detector matrix channels:**
- Channel 0: electron/photon densities
- Channel 1: muon densities
- (Arrival time channel is dropped — it is unstable between simulation and real data)

**5 reconstructed features** (stable across simulation and real data):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | E | log10(reconstructed energy / eV) |
| 1 | Ze | Zenith angle (degrees) |
| 2 | Az | Azimuth angle (degrees) |
| 3 | Ne | log10(electron number) |
| 4 | Nmu | log10(muon number) |

When flattened: 16×16×2 + 5 = **517 dimensions**.

### Raw data files

The raw files contain more columns than the standard input. You can use the standard input or go back to the raw data.

**Matrix data** (NPZ format, two hadronic interaction models):
- `{mode}_matrices.npz` — key: `matrices`, shape: `(N, 16, 16, 3)` (3rd channel is arrival time)
- `{mode}_features.npz` — key: `features`, shape: `(N, 10)`
- `{mode}_true_features.npz` — key: `true_features`, shape: `(N, 10)`

Where `mode` is `qgs_spectra` (4.16M events) or `epos_spectra` (2.79M events).

**Reconstructed features** (all 10 columns — indices 2-4, 9 are unstable across sim/real):
```
features[:, 0]  = particle type (reconstructed) — DO NOT use as model input
features[:, 1]  = E       — log10(energy/eV), reconstructed
features[:, 2]  = Xc      — shower core X position (unstable)
features[:, 3]  = Yc      — shower core Y position (unstable)
features[:, 4]  = core_dist — distance from array center (unstable)
features[:, 5]  = Ze      — zenith angle (degrees)
features[:, 6]  = Az      — azimuth angle (degrees)
features[:, 7]  = Ne      — log10(electron number)
features[:, 8]  = Nmu     — log10(muon number)
features[:, 9]  = Age     — shower age parameter (unstable)
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

**Tabular data** (CSV, alternative representation):
`data_array_{epos-LHC,qgs-4,sibyll-23c}.csv` — columns: `E, Xc, Yc, Ze, Az, Ne, Nmu, Age, p_idx, hadron`

### Train/test split

Each task has a fixed 80/20 random split (seed=2026). **Do not modify the splits.**
- Composition: split across QGSJet-II + EPOS-LHC spectra data
- Gamma: split across QGSJet-II + EPOS-LHC + SIBYLL gamma+proton data

## Quality Cuts

Previous published analyses applied these cuts. You may choose to apply them or not — this is a modeling decision:
- Ze < 18 degrees (zenith angle)
- Ne > 4.8 (log10 electron number)
- Nmu > 3.6 (log10 muon number)
- 0.2 < Age < 1.48 (shower age)

## What to Submit

Your submission directory must contain:
1. **`predictions.npz`** with one or both keys:
   - `predictions`: int array of class labels (0-4) for each test sample — for Task 1
   - `gamma_scores`: float array of gamma probabilities (higher = more gamma-like) — for Task 2
2. **`README.md`** — document your approach: what you tried, what failed, what worked, architecture details, training curves, anything interesting you found
3. **All source code** needed to reproduce the result

### If your solution was built by an AI agent, also include:
- **`metadata.yaml`** with: agent model, prompt/config used, total cost ($), token count, wall-clock time, number of iterations/experiments
- **Full reasoning trace** if available (conversation log, git history of experiments)

## Evaluation

```bash
# Task 1: Mass composition
python verify.py submissions/your_submission/predictions.npz

# Task 2: Gamma/hadron
python verify.py --task gamma submissions/your_submission/predictions.npz
```

## Baselines

**Mass composition (Task 1):**
- Random: ~20% accuracy
- Dense NN (SELU + AlphaDropout, 1.8M params, flattened matrices + Ze, Az): ~44.5% accuracy
- Dense NN (ELU + BatchNorm, 798K params): ~47% accuracy
- Ensemble of both: slightly higher
- Published (JCAP 2024, years of expert iteration): the target to beat

**Gamma/hadron (Task 2):**
- Published (Chinese Physics C 2023): hadronic survival rate 10⁻⁶ to 3×10⁻⁵ at 0.3-10 PeV

## Physics Background

When a high-energy cosmic ray hits Earth's atmosphere, it creates a cascade of secondary particles called an extensive air shower. KASCADE measured these showers with a 200×200m array of 252 detector stations in Karlsruhe, Germany, over ~25 years.

The classification task is hard because:
- Air showers are inherently stochastic — the same primary particle can produce very different showers
- The mapping from observables to primary particle is many-to-one and noisy
- Different mass groups overlap significantly in observable space, especially at lower energies
- The hadronic interaction models (QGSJet, EPOS) used in simulation don't perfectly match real data

Light particles (protons) produce deeper, more variable showers with fewer muons. Heavy particles (iron) produce shallower, more regular showers with more muons. The electron-to-muon ratio (Ne/Nmu) is the strongest discriminant, but it's far from sufficient alone.
