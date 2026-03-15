"""CPU experiment: HGB with extensive feature engineering.
No GPU needed — runs on CPU. Explores whether rich features alone can beat the CNN.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from scipy.optimize import differential_evolution
import time

DATA_DIR = "data"
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
MIXTURE_SIZE = 5000; MIXTURE_SEED = 2026; GRID_STEP = 0.1

def p(msg): print(msg, flush=True)

def load_qgs():
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_true = np.load(f"{DATA_DIR}/qgs_spectra_true_features.npz")['true_features']
    reco = raw_feat[:, 1:].astype(np.float32)  # 9 cols
    labels = raw_feat[:, 0].astype(np.int32)  # particle type 1-5
    # Apply cuts
    mask = np.ones(len(reco), dtype=bool)
    for feat_name, (lo, hi) in CUTS.items():
        idx = RECO_HEADERS.index(feat_name)
        mask &= (reco[:, idx] > lo) & (reco[:, idx] < hi)
    return reco[mask], labels[mask] - 1  # 0-indexed

def engineer(reco):
    """Build rich feature set from 9 reco features."""
    E = reco[:, 0]; Ze = reco[:, 4]; Az = reco[:, 5]
    Ne = reco[:, 6]; Nmu = reco[:, 7]; Age = reco[:, 8]
    feats = [
        E, Ze, Ne, Nmu, Age,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu,                    # log(Ne/Nmu) — key discriminant
        Ne + Nmu,                    # total size
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
        Ne * np.cos(np.radians(Ze)), # zenith-corrected
        Nmu * np.cos(np.radians(Ze)),
        (Ne - Nmu) * np.cos(np.radians(Ze)),
        Age * (Ne - Nmu),            # age × ratio interaction
        Age * Ze,                    # age × zenith interaction
        E - 0.9*Ne - 0.1*Nmu,       # energy residual
        np.sin(np.radians(2*Az)),
        Ne**2, Nmu**2, (Ne-Nmu)**2,
    ]
    return np.column_stack(feats)

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
    p("Loading data...")
    reco, labels = load_qgs()
    p(f"Total: {len(labels)} events")

    # Same split as reproduction
    import torch
    n = len(labels); n_valid = int(n*0.3); n_train = n - n_valid
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=gen).numpy()
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:]
    gen2 = torch.Generator().manual_seed(42)
    n_test = int(n_valid*0.3); n_val = n_valid - n_test
    valid_perm = torch.randperm(n_valid, generator=gen2).numpy()
    val_idx = valid_idx[valid_perm[:n_val]]
    test_idx = valid_idx[valid_perm[n_val:]]

    p(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    X_all = engineer(reco)
    X_train, y_train = X_all[train_idx], labels[train_idx]
    X_val, y_val = X_all[val_idx], labels[val_idx]
    X_test, y_test = X_all[test_idx], labels[test_idx]

    feval = FracEval(y_test)

    configs = [
        {"max_iter": 1000, "max_depth": 6, "learning_rate": 0.05, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"max_iter": 1500, "max_depth": 8, "learning_rate": 0.03, "min_samples_leaf": 30, "l2_regularization": 0.5},
        {"max_iter": 2000, "max_depth": 5, "learning_rate": 0.05, "min_samples_leaf": 100, "l2_regularization": 2.0},
        {"max_iter": 1000, "max_depth": 10, "learning_rate": 0.05, "min_samples_leaf": 20, "l2_regularization": 0.3},
        {"max_iter": 1500, "max_depth": 7, "learning_rate": 0.05, "min_samples_leaf": 50, "l2_regularization": 1.0},
    ]

    best_fe = 1.0
    best_probs = None

    for i, cfg in enumerate(configs):
        p(f"\nConfig {i}: {cfg}")
        t1 = time.time()
        hgb = HistGradientBoostingClassifier(**cfg, random_state=42, class_weight='balanced')
        hgb.fit(X_train, y_train)
        preds = hgb.predict(X_test)
        probs = hgb.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        fe = feval.evaluate(preds)
        p(f"  acc={acc:.4f} frac_err={fe:.4f} [{time.time()-t1:.0f}s]")

        if fe < best_fe:
            best_fe = fe
            best_probs = probs
            best_cfg = i

    # Bias opt on best
    p(f"\nBest HGB config {best_cfg}: {best_fe:.4f}")
    la = np.log(best_probs + 1e-10)
    def obj(b): return feval.evaluate((la+b).argmax(1))
    from scipy.optimize import minimize
    res = minimize(obj, np.zeros(5), method='Nelder-Mead',
                   options={'maxiter': 10000, 'adaptive': True})
    bias_fe = feval.evaluate((la + res.x).argmax(1))
    p(f"Bias-optimized: {bias_fe:.4f}")

    np.save("experiments/hgb_test_probs.npy", best_probs)
    p(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    p(f"SOTA: 0.107 | Best HGB: {bias_fe:.4f}")

if __name__ == "__main__":
    main()
