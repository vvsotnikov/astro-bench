"""CPU experiment: Rich tabular features from matrices + reco features.
Extract spatial statistics from detector matrices and combine with engineered reco features.
No CNN — pure tabular ML. Provides a fundamentally different error pattern for ensembling.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import random_split
import time

DATA_DIR = "data"
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
MIXTURE_SIZE = 5000; MIXTURE_SEED = 2026; GRID_STEP = 0.1

def p(msg): print(msg, flush=True)

def load_and_split():
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32)
    labels = raw_feat[:, 0].astype(np.int32) - 1  # 0-indexed
    # Apply cuts
    mask = np.ones(len(reco), dtype=bool)
    for feat_name, (lo, hi) in CUTS.items():
        idx = RECO_HEADERS.index(feat_name)
        mask &= (reco[:, idx] > lo) & (reco[:, idx] < hi)
    reco = reco[mask]; labels = labels[mask]; mat = raw_mat[mask]
    # Same split
    n = len(labels); nv = int(n*0.3); nt = n - nv
    idx_all = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    train_idx = idx_all[:nt]; valid_idx = idx_all[nt:]
    ntest = int(nv*0.3); nval = nv - ntest
    vperm = torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    val_idx = valid_idx[vperm[:nval]]; test_idx = valid_idx[vperm[nval:]]
    return reco, labels, mat, train_idx, val_idx, test_idx

def extract_matrix_features(mat):
    """Extract rich spatial statistics from 16x16x3 matrices."""
    n = len(mat)
    ch_e = mat[:, :, :, 1].astype(np.float32)  # electron/photon channel
    ch_m = mat[:, :, :, 2].astype(np.float32)  # muon channel

    feats = []
    yy, xx = np.mgrid[0:16, 0:16]

    for ch, name in [(ch_e, 'e'), (ch_m, 'mu')]:
        total = ch.sum(axis=(1, 2))
        max_val = ch.max(axis=(1, 2))
        nnz = (ch > 0).sum(axis=(1, 2)).astype(np.float32)
        mean_nz = np.where(nnz > 0, total / (nnz + 1e-6), 0)

        # Center of mass
        total_safe = np.where(total > 0, total, 1)
        cx = (ch * xx[None]).sum(axis=(1, 2)) / total_safe
        cy = (ch * yy[None]).sum(axis=(1, 2)) / total_safe

        # Spread (weighted std)
        spread_x = np.sqrt(np.abs((ch * (xx[None] - cx[:, None, None])**2).sum(axis=(1, 2)) / total_safe))
        spread_y = np.sqrt(np.abs((ch * (yy[None] - cy[:, None, None])**2).sum(axis=(1, 2)) / total_safe))

        # Kurtosis
        r2 = (xx[None] - cx[:, None, None])**2 + (yy[None] - cy[:, None, None])**2
        m4 = (ch * r2**2).sum(axis=(1, 2)) / total_safe
        m2 = (ch * r2).sum(axis=(1, 2)) / total_safe
        m2_safe = np.where(m2 > 0, m2, 1)
        kurtosis = m4 / m2_safe**2

        # Percentiles of non-zero values
        # Use quantiles on sorted non-zero values
        p25 = np.zeros(n); p75 = np.zeros(n)
        for j in range(n):
            nz = ch[j][ch[j] > 0]
            if len(nz) > 0:
                p25[j] = np.percentile(nz, 25)
                p75[j] = np.percentile(nz, 75)

        # Quadrant sums (asymmetry)
        q1 = ch[:, :8, :8].sum(axis=(1, 2))
        q2 = ch[:, :8, 8:].sum(axis=(1, 2))
        q3 = ch[:, 8:, :8].sum(axis=(1, 2))
        q4 = ch[:, 8:, 8:].sum(axis=(1, 2))
        asym_x = (q1 + q3 - q2 - q4) / (total_safe + 1e-6)
        asym_y = (q1 + q2 - q3 - q4) / (total_safe + 1e-6)

        feats.extend([
            np.log1p(total), np.log1p(max_val), nnz / 256,
            np.log1p(mean_nz),
            cx / 15, cy / 15,
            spread_x / 8, spread_y / 8,
            kurtosis,
            np.log1p(p25), np.log1p(p75),
            asym_x, asym_y,
        ])

    # Cross-channel features
    total_e = ch_e.sum(axis=(1, 2))
    total_m = ch_m.sum(axis=(1, 2))
    feats.append(np.log1p(total_e) - np.log1p(total_m))  # e/mu ratio
    feats.append((ch_e > 0).sum(axis=(1, 2)).astype(np.float32) /
                 ((ch_m > 0).sum(axis=(1, 2)).astype(np.float32) + 1))  # nnz ratio

    return np.column_stack(feats)

def engineer_reco(reco):
    E = reco[:, 0]; Ze = reco[:, 4]; Az = reco[:, 5]
    Ne = reco[:, 6]; Nmu = reco[:, 7]; Age = reco[:, 8]
    return np.column_stack([
        E, Ze, Ne, Nmu, Age,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu,
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
        Ne * np.cos(np.radians(Ze)),
        Nmu * np.cos(np.radians(Ze)),
        (Ne - Nmu) * np.cos(np.radians(Ze)),
        Age * (Ne - Nmu),
        Age * Ze,
        E - 0.9*Ne - 0.1*Nmu,
        np.sin(np.radians(2*Az)),
        Ne**2, Nmu**2, (Ne-Nmu)**2,
        Age**2, Ze**2,
    ])

def gen_grid(n=5, step=0.1):
    ns=round(1.0/step); f=[]
    def r(rem,d,c):
        if d==n-1: c.append(rem*step); f.append(c[:]); c.pop(); return
        for i in range(rem+1): c.append(i*step); r(rem-i,d+1,c); c.pop()
    r(ns,0,[]); return np.array(f)

class FracEval:
    def __init__(self, y):
        ci={c:np.where(y==c)[0] for c in range(5)}
        self.fracs=gen_grid(); self.ne=len(self.fracs)
        rng=np.random.default_rng(MIXTURE_SEED)
        self.si=[]; self.tf=np.zeros((self.ne,5))
        for i in range(self.ne):
            counts=np.round(self.fracs[i]*MIXTURE_SIZE).astype(int)
            d=MIXTURE_SIZE-counts.sum()
            if d: counts[np.argmax(counts)]+=d
            idx=[]
            for c in range(5):
                if counts[c]<=0: continue
                idx.append(rng.choice(ci[c],size=counts[c],replace=True))
                self.tf[i,c]=counts[c]/MIXTURE_SIZE
            self.si.append(np.concatenate(idx))
    def evaluate(self, p):
        e=np.zeros((self.ne,5))
        for i in range(self.ne):
            s=p[self.si[i]]; pc=np.bincount(s,minlength=5)[:5]
            e[i]=np.abs(self.tf[i]-pc/pc.sum())
        return float(e.mean())

def main():
    t0 = time.time()
    p("Loading and splitting data...")
    reco, labels, mat, train_idx, val_idx, test_idx = load_and_split()
    p(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    p("Extracting matrix features...")
    mat_feats = extract_matrix_features(mat)
    p(f"Matrix features: {mat_feats.shape[1]}")

    reco_feats = engineer_reco(reco)
    p(f"Reco features: {reco_feats.shape[1]}")

    X = np.concatenate([reco_feats, mat_feats], axis=1)
    p(f"Total features: {X.shape[1]}")

    X_train, y_train = X[train_idx], labels[train_idx]
    X_test, y_test = X[test_idx], labels[test_idx]

    feval = FracEval(y_test)

    configs = [
        ("HGB d=8 lr=0.03", HistGradientBoostingClassifier(
            max_iter=2000, max_depth=8, learning_rate=0.03, min_samples_leaf=30,
            l2_regularization=0.5, random_state=42, class_weight='balanced')),
        ("HGB d=10 lr=0.05", HistGradientBoostingClassifier(
            max_iter=1500, max_depth=10, learning_rate=0.05, min_samples_leaf=20,
            l2_regularization=0.3, random_state=42, class_weight='balanced')),
        ("HGB d=6 lr=0.05 big", HistGradientBoostingClassifier(
            max_iter=3000, max_depth=6, learning_rate=0.05, min_samples_leaf=50,
            l2_regularization=1.0, random_state=42, class_weight='balanced')),
        ("RF 1000", RandomForestClassifier(
            n_estimators=1000, max_depth=20, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1)),
    ]

    for name, clf in configs:
        t1 = time.time()
        p(f"\n{name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        fe = feval.evaluate(preds)
        p(f"  acc={acc:.4f} fe={fe:.4f} [{time.time()-t1:.0f}s]")
        np.save(f"experiments/tabular_{name.replace(' ', '_')}_probs.npy", probs)

    p(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
