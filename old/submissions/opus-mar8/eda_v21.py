"""EDA: Deep analysis of test set confusion patterns, especially helium.
Also analyze train vs test distribution shift.
"""
import numpy as np
import json

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
NAMES = ["proton", "helium", "carbon", "silicon", "iron"]

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    return np.stack([
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu, (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
    ], axis=1).astype(np.float32)

# Load test data
p("=== TEST SET ANALYSIS ===")
feat_test = np.load(f"{DATA_DIR}/composition_test/features.npy")
y_test = np.load(f"{DATA_DIR}/composition_test/labels_composition.npy")
probs_v8 = np.load(f"{OUT_DIR}/probs_v8.npy")
preds_v8 = probs_v8.argmax(1)

E, Ze, Az, Ne, Nmu = feat_test[:, 0], feat_test[:, 1], feat_test[:, 2], feat_test[:, 3], feat_test[:, 4]

# Class distribution
p("\n--- Class distribution (test) ---")
for c in range(5):
    n = (y_test == c).sum()
    p(f"  {NAMES[c]:8s}: {n:6d} ({n/len(y_test)*100:.1f}%)")

# Per-class feature distributions
p("\n--- Feature distributions by class (test) ---")
feat_names = ["E", "Ze", "Az", "Ne", "Nmu", "Ne-Nmu", "Ne/Nmu"]
for c in range(5):
    mask = y_test == c
    p(f"\n  {NAMES[c]}:")
    for fi, fn in enumerate(["E", "Ze", "Az", "Ne", "Nmu"]):
        vals = feat_test[mask, fi]
        p(f"    {fn:4s}: mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f}")
    ratio = (Ne[mask] - Nmu[mask])
    p(f"    Ne-Nmu: mean={ratio.mean():.3f} std={ratio.std():.3f}")

# Confusion matrix analysis with v8 predictions
p("\n--- Confusion matrix (v8, row-normalized) ---")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, preds_v8)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
header = "True\\Pred  " + "  ".join(f"{n[:4]:>6}" for n in NAMES)
p(header)
for i in range(5):
    row = "  ".join(f"{cm_norm[i,j]:>6.3f}" for j in range(5))
    p(f"  {NAMES[i][:4]:<8}  {row}")

# Where helium goes wrong
p("\n--- Helium misclassification analysis ---")
he_mask = y_test == 1  # true helium
he_correct = (preds_v8 == 1) & he_mask
he_as_proton = (preds_v8 == 0) & he_mask
he_as_carbon = (preds_v8 == 2) & he_mask

p(f"  Helium total: {he_mask.sum()}")
p(f"  Correct: {he_correct.sum()} ({he_correct.sum()/he_mask.sum()*100:.1f}%)")
p(f"  -> proton: {he_as_proton.sum()} ({he_as_proton.sum()/he_mask.sum()*100:.1f}%)")
p(f"  -> carbon: {he_as_carbon.sum()} ({he_as_carbon.sum()/he_mask.sum()*100:.1f}%)")

# Feature comparison: correctly classified He vs He->proton
p("\n  Features of He correctly classified vs He misclassified as proton:")
for fi, fn in enumerate(["E", "Ze", "Az", "Ne", "Nmu"]):
    correct_vals = feat_test[he_correct, fi]
    wrong_vals = feat_test[he_as_proton, fi]
    p(f"    {fn:4s}: correct={correct_vals.mean():.3f}+/-{correct_vals.std():.3f}  "
      f"->proton={wrong_vals.mean():.3f}+/-{wrong_vals.std():.3f}")

# Energy-binned accuracy
p("\n--- Energy-binned accuracy (v8) ---")
energy_bins = [(14.0, 15.0), (15.0, 15.5), (15.5, 16.0), (16.0, 16.5), (16.5, 17.0), (17.0, 18.0)]
for lo, hi in energy_bins:
    mask = (E >= lo) & (E < hi)
    if mask.sum() == 0:
        continue
    acc = (preds_v8[mask] == y_test[mask]).mean()
    p(f"  E=[{lo:.1f},{hi:.1f}): n={mask.sum():5d}, acc={acc:.4f}")
    # Per-class accuracy in this bin
    for c in range(5):
        cmask = mask & (y_test == c)
        if cmask.sum() > 0:
            cacc = (preds_v8[cmask] == y_test[cmask]).mean()
            p(f"    {NAMES[c]:8s}: n={cmask.sum():5d}, acc={cacc:.4f}")

# Probability confidence analysis
p("\n--- Prediction confidence analysis ---")
max_prob = probs_v8.max(1)
for c in range(5):
    mask = y_test == c
    correct = preds_v8[mask] == c
    p(f"  {NAMES[c]:8s}: avg_conf_correct={max_prob[mask & (preds_v8==c)].mean():.4f} "
      f"avg_conf_wrong={max_prob[mask & (preds_v8!=c)].mean():.4f}")

# Train vs test distribution comparison
p("\n\n=== TRAIN vs TEST DISTRIBUTION ===")
feat_train = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
y_train = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')

# Sample train for speed
idx = np.random.default_rng(42).choice(len(y_train), size=500000, replace=False)
idx.sort()

p("\n--- Class distribution (train sample 500K) ---")
yt_sample = np.array(y_train[idx])
for c in range(5):
    n = (yt_sample == c).sum()
    p(f"  {NAMES[c]:8s}: {n:6d} ({n/len(yt_sample)*100:.1f}%)")

p("\n--- Feature distributions (train vs test) ---")
ft_sample = np.array(feat_train[idx], dtype=np.float32)
for fi, fn in enumerate(["E", "Ze", "Az", "Ne", "Nmu"]):
    tr = ft_sample[:, fi]
    te = feat_test[:, fi]
    p(f"  {fn:4s}: train=[{tr.mean():.3f}+/-{tr.std():.3f}] test=[{te.mean():.3f}+/-{te.std():.3f}]")

# What fraction of train passes test quality cuts?
# Test cuts: Ze<30, Ne>4.8 (no Age available)
Ze_train = ft_sample[:, 1]
Ne_train = ft_sample[:, 3]
passes_cuts = (Ze_train < 30) & (Ne_train > 4.8)
p(f"\n  Train passing test cuts (Ze<30, Ne>4.8): {passes_cuts.sum()}/{len(ft_sample)} = {passes_cuts.mean()*100:.1f}%")

# Matrix sparsity
p("\n--- Matrix sparsity analysis (test) ---")
mat_test = np.load(f"{DATA_DIR}/composition_test/matrices.npy", mmap_mode='r')
# Check first 10K
sample_mat = np.array(mat_test[:10000], dtype=np.float32)
ch0_sparsity = (sample_mat[:,:,:,0] == 0).mean()
ch1_sparsity = (sample_mat[:,:,:,1] == 0).mean()
p(f"  Channel 0 (electron) zeros: {ch0_sparsity*100:.1f}%")
p(f"  Channel 1 (muon) zeros: {ch1_sparsity*100:.1f}%")

# Per-class matrix statistics
for c in range(5):
    mask = y_test[:10000] == c
    if mask.sum() == 0:
        continue
    m = sample_mat[mask]
    ch0_sum = m[:,:,:,0].sum(axis=(1,2)).mean()
    ch1_sum = m[:,:,:,1].sum(axis=(1,2)).mean()
    ch0_max = m[:,:,:,0].max(axis=(1,2)).mean()
    ch1_max = m[:,:,:,1].max(axis=(1,2)).mean()
    nz0 = (m[:,:,:,0] > 0).sum(axis=(1,2)).mean()
    nz1 = (m[:,:,:,1] > 0).sum(axis=(1,2)).mean()
    p(f"  {NAMES[c]:8s}: e_sum={ch0_sum:.1f} m_sum={ch1_sum:.1f} "
      f"e_max={ch0_max:.1f} m_max={ch1_max:.1f} e_nz={nz0:.1f} m_nz={nz1:.1f}")

p("\n--- DONE ---")
