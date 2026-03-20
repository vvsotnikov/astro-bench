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

**Key metric: mean fraction error** (lower is better). This measures how well the classifier recovers true particle fractions across random mixture compositions, following the methodology of Kuznetsov et al. (JINST 2024, Section 4.3, Table 1). All 5-component fraction combinations on a 0.1 step grid summing to 1.0 are enumerated (1001 grid points). For each, an ensemble of 5000 events is sampled from the test set with those fractions, and the absolute error |true_fraction - predicted_fraction| is computed per class. The metric is the mean across all classes and grid points (fixed seed=2026). A perfect classifier scores 0. This directly measures what matters for physics: the accuracy of unfolded mass composition spectra.

### Task 2: Gamma/Hadron Separation (binary classification)

Distinguish gamma-ray primaries from hadronic cosmic rays (protons in simulation).

| Class | Type |
|-------|------|
| 0 | Gamma |
| 1 | Hadron (proton) |

In real data, the gamma-to-hadron ratio is ~1:1,000,000. In the simulation test set, it's ~1:23. Evaluation must account for extreme class imbalance in deployment.

**Key metric: hadronic survival rate at 75% gamma efficiency** (i.e., what fraction of hadrons survive your cut while keeping 75% of gammas). Lower is better. Published baseline: suppression power 10²–10³ at ~30–70% gamma efficiency using RF regressor (Kostunin et al., [ICRC 2021](https://arxiv.org/abs/2108.03407)).

## The Data

All data is in `./data/` after running `python download_data.py`.

### Standard input format

Each task has its own pre-split dataset. All files are `.npy` format and support memory-mapping for zero-copy access:

```python
X = np.load('data/composition_train/matrices.npy', mmap_mode='r')  # instant, zero RAM
```

**Task 1 — Mass composition:** `data/composition_train/` and `data/composition_test/`
- Source: QGSJet-II.04 simulation (~4.2M raw events → 383K after quality cuts)
- Quality cuts pre-applied: Ze<30, Ne>4.8, Age∈(0.2, 1.48), Nmu>3.6
- Split: 70% train (268K) / 30% test (115K), seed=42
- Matches published pipeline (Kuznetsov et al., JINST 2024) exactly

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `matrices.npy` | `(N, 16, 16, 2)` | float32 | Detector grid images |
| `features.npy` | `(N, 6)` | float32 | Reconstructed scalar features |
| `labels_composition.npy` | `(N,)` | int8 | Mass class (0-4) |

**Task 2 — Gamma/hadron:** `data/gamma_train/` and `data/gamma_test/`
- Sources: QGSJet-II + EPOS-LHC + SIBYLL gamma+proton simulations (~1.9M events total, ~91K gamma + ~1.8M proton)
- Train: ~1.5M events (no quality cuts), Test: ~36K events (quality cuts applied, ~1.5K gamma)

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `matrices.npy` | `(N, 16, 16, 2)` | float32 | Detector grid images |
| `features.npy` | `(N, 5)` | float32 | Reconstructed scalar features |
| `labels_gamma.npy` | `(N,)` | int8 | 0=gamma, 1=hadron |

### Detector matrices

Each 16×16 grid represents the KASCADE detector array (200×200m, 252 stations binned into 16×16 cells).

**Two channels** (after dropping the unstable arrival time channel):
- Channel 0: electron/photon densities — median ~100, long tail up to ~5000
- Channel 1: muon densities — median ~50, long tail up to ~600
- ~85% of grid cells are zero (sparse)

The raw 3-channel matrices (with arrival time as channel 0) are available in the NPZ files if you want to experiment, but arrival time distributions differ between simulation and real KASCADE data.

### Reconstructed features

**Composition task — 6 features:**

| Index | Feature | Description | Typical range |
|-------|---------|-------------|---------------|
| 0 | E | log10(reconstructed energy / eV) | 14–18 |
| 1 | Ze | Zenith angle (degrees) | 0–30 |
| 2 | Az | Azimuth angle (degrees) | 0–360 |
| 3 | Ne | log10(electron number) | 4.8–7.5 |
| 4 | Nmu | log10(muon number) | 3.6–6.5 |
| 5 | Age | Shower age parameter | 0.2–1.48 |

The published baseline (JINST 2024) uses Ne, Nmu, Age, and Ze as model inputs with the following normalization:
- Ne: `(Ne - 5.31) / 0.5`
- Nmu: `(Nmu - 4.3) / 0.42`
- Age: `Age - 1.0`
- Ze: `Ze / 60.0`

**Gamma task — 5 features** (E, Ze, Az, Ne, Nmu — no Age, no Nmu cut).

When flattened: 16×16×2 + 6 = **518 dimensions** (composition) or 16×16×2 + 5 = **517 dimensions** (gamma).

### Physics of the features

Understanding what the features mean physically can inform better model design:

- **Ne/Nmu ratio** is the single strongest discriminant for mass composition. Iron nuclei produce ~3× more muons than protons at the same energy, because iron showers develop as 56 independent sub-showers. This is why flattened-matrix + Ne + Nmu approaches work well.

- **Gamma rays** produce purely electromagnetic showers — almost no muons (median log10(Nmu) ≈ 3.0 vs 3.5 for protons). This is the fundamental physics signature for gamma/hadron separation. A good gamma classifier should heavily leverage the muon channel.

- **Zenith angle (Ze)** affects shower development — inclined showers traverse more atmosphere, producing different Ne/Nmu profiles at ground level.

- **Energy (E)** is reconstructed from Ne and Nmu, so it's not independent. At higher energies (>10^16 eV), mass separation improves because shower fluctuations become relatively smaller.

- **Azimuth (Az)** has weak discriminating power but encodes geomagnetic effects on shower development.

### Raw data files

The raw files contain more columns than the standard input. You can use the standard input or go back to the raw data.

**Matrix data** (NPZ format):
- `{mode}_matrices.npz` — key: `matrices`, shape: `(N, 16, 16, 3)` (channel 0 is arrival time)
- `{mode}_features.npz` — key: `features`, shape: `(N, 10)`
- `{mode}_true_features.npz` — key: `true_features`, shape: `(N, 10)`

Where `mode` is:
- Composition: `qgs_spectra` (4.16M events) or `epos_spectra` (2.79M events)
- Gamma: `qgs-4_gm_pr` (634K), `LHC_gm_pr` (628K), or `sibyll-23c_gm_pr` (652K)

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

**Composition:** 70/30 random split (seed=42) on QGSJet-II.04 data after quality cuts. Matches the published pipeline exactly. **Do not modify the splits.**

**Gamma:** 80/20 random split (seed=2026) across QGSJet-II + EPOS-LHC + SIBYLL. **Do not modify the splits.**

## Quality Cuts

**Composition task** — all cuts are pre-applied to both train and test data:
- **Ze < 30** degrees (zenith angle)
- **Ne > 4.8** (log10 electron number)
- **0.2 < Age < 1.48** (shower age)
- **Nmu > 3.6** (log10 muon number)

These match the published analysis (Kuznetsov et al., JINST 2024).

**Gamma task** — cuts pre-applied to test data only:
- **Ze < 30**, **Ne > 4.8**, **0.2 < Age < 1.48**
- No Nmu cut (gamma showers produce almost no muons — that's the signal)
- Training data has no cuts — apply them as a modeling decision.

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
- Random: ~0.20 fraction error
- RandomForest (6 features only): ~0.11 fraction error
- Published CNN (LeNet, 16×16×2 + Ne, Nmu, Age, Ze, ~36K params): fraction error 0.107, ~51% accuracy ([JINST 2024](https://arxiv.org/abs/2311.06893))
- Same data, same splits, same evaluation — your model is directly comparable

**Gamma/hadron (Task 2):**
- Published ([ICRC 2021](https://arxiv.org/abs/2108.03407)): hadron suppression power 10²–10³ at ~30–70% gamma efficiency using RF regressor on scalar features

## Physics Background

When a high-energy cosmic ray hits Earth's atmosphere, it creates a cascade of secondary particles called an **extensive air shower** (EAS). KASCADE measured these showers with a 200×200m array of 252 detector stations in Karlsruhe, Germany, over ~25 years (1996-2013).

Each detector station measured:
- **Electron/photon density** — from the electromagnetic component of the shower
- **Muon density** — from the hadronic component (muons penetrate deeper due to higher mass)
- **Arrival time** — when the shower front passed the station (not used here — unstable between sim/real)

The 16×16 grid is a binned representation of the 252 stations. Most cells are zero because the shower core doesn't illuminate the entire array.

### Why classification is hard

- **Stochastic showers**: the same primary particle can produce very different showers depending on where the first interaction occurs
- **Many-to-one mapping**: different primaries can produce identical-looking showers
- **Mass overlap**: neighboring mass groups (H/He, Si/Fe) overlap significantly, especially at lower energies
- **Hadronic model uncertainty**: QGSJet-II and EPOS-LHC (the two simulation codes) predict different shower properties for the same primary — your model trains on both, so it must generalize across model differences
- **Energy dependence**: classification accuracy improves significantly at higher energies (>10^16 eV) because shower fluctuations become relatively smaller

### Why this matters

The mass composition of cosmic rays at PeV energies is one of the key open questions in astroparticle physics. The "knee" in the cosmic ray spectrum (~3×10^15 eV) is believed to mark the energy where galactic cosmic ray sources reach their maximum, but the exact composition tells us whether this is due to rigidity-dependent acceleration limits or propagation effects. Better classifiers enable better unfolding of the true composition from measured data.

For gamma rays: detecting PeV gamma-ray sources ("PeVatrons") would identify the sites where cosmic rays are accelerated to knee energies. The challenge is that for every gamma ray, there are ~10^6 hadronic cosmic rays — your classifier must achieve extreme background rejection while maintaining high gamma efficiency.
