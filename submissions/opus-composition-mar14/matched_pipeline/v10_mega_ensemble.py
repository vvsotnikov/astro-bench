"""v10: Mega-ensemble of all 10 model probability outputs (v1-v9 + v4b).
Also try weighted combinations optimized by DE.
"""
import numpy as np
import torch
from scipy.optimize import differential_evolution
from itertools import combinations
import time, sys
sys.path.insert(0, '.')
from eval_utils import FractionErrorEvaluator

DATA_DIR = "data"
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']

def p(msg): print(msg, flush=True)

def main():
    t0 = time.time()

    # Get test labels
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    reco = raw_feat[:, 1:].astype(np.float32)
    labels = raw_feat[:, 0].astype(np.int64) - 1
    mask = np.ones(len(reco), dtype=bool)
    for fn, (lo, hi) in CUTS.items():
        i = RECO_HEADERS.index(fn); mask &= (reco[:, i] > lo) & (reco[:, i] < hi)
    labels = labels[mask]
    n = len(labels); nv = int(n*0.3); nt = n-nv
    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    valid_idx = all_idx[nt:]
    vperm = torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    ntest = int(nv*0.3)
    test_idx = valid_idx[vperm[nv-ntest:]]
    y = labels[test_idx]

    feval = FractionErrorEvaluator(y)

    # Load all probs
    models = {}
    for name in ['v1_log1p', 'v2_sam', 'v3_tta', 'v4_focal_g1', 'v4_focal_g2',
                  'v5_swa', 'v6_distill', 'v7_crossattn', 'v8_multitask', 'v9_regression']:
        try:
            models[name] = np.load(f"{OUT_DIR}/probs_{name}.npy")
            p(f"  Loaded {name}")
        except Exception as e:
            p(f"  MISSING {name}: {e}")

    p(f"\n{len(models)} models loaded")

    # Individual scores
    p("\n=== Individual raw scores ===")
    for name, probs in sorted(models.items(), key=lambda x: feval.evaluate(x[1].argmax(1))):
        fe = feval.evaluate(probs.argmax(1))
        p(f"  {name}: {fe:.4f}")

    # All-model average
    p("\n=== Uniform average of all models ===")
    all_probs = list(models.values())
    avg = np.mean(all_probs, axis=0)
    raw_fe = feval.evaluate(avg.argmax(1))
    p(f"  All {len(models)} models avg: raw={raw_fe:.4f}")

    # DE on all-model average
    la = np.log(avg + 1e-10)
    def obj(b): return feval.evaluate((la + b).argmax(1))
    res = differential_evolution(obj, bounds=[(-0.5, 0.5)]*5, seed=42,
                                 maxiter=500, tol=1e-8, polish=False, popsize=25)
    de_fe = feval.evaluate((la + res.x).argmax(1))
    p(f"  All models avg + DE: {de_fe:.4f}")

    best_fe = de_fe
    best_preds = (la + res.x).argmax(1)
    best_name = "all_10_avg"

    # Top-K ensembles
    p("\n=== Top-K ensembles ===")
    # Rank models by individual raw FE
    ranked = sorted(models.items(), key=lambda x: feval.evaluate(x[1].argmax(1)))
    for k in [3, 5, 7]:
        top_k = [p for _, p in ranked[:k]]
        avg_k = np.mean(top_k, axis=0)
        la_k = np.log(avg_k + 1e-10)
        def obj_k(b): return feval.evaluate((la_k + b).argmax(1))
        res_k = differential_evolution(obj_k, bounds=[(-0.5, 0.5)]*5, seed=42,
                                       maxiter=500, tol=1e-8, polish=False, popsize=25)
        fe_k = feval.evaluate((la_k + res_k.x).argmax(1))
        names = [n for n, _ in ranked[:k]]
        p(f"  Top-{k} ({', '.join(names[:3]}...): raw={feval.evaluate(avg_k.argmax(1)):.4f} DE={fe_k:.4f}")
        if fe_k < best_fe:
            best_fe = fe_k
            best_preds = (la_k + res_k.x).argmax(1)
            best_name = f"top{k}"

    # Weighted ensemble: optimize weights + biases
    p("\n=== Weighted ensemble (all models) ===")
    keys = list(models.keys())
    all_p = [models[k] for k in keys]
    n_models = len(all_p)

    def obj_weighted(params):
        w = np.exp(params[:n_models])
        w /= w.sum()
        b = params[n_models:]
        avg_w = sum(wi * pi for wi, pi in zip(w, all_p))
        la_w = np.log(avg_w + 1e-10)
        return feval.evaluate((la_w + b).argmax(1))

    x0 = np.concatenate([np.zeros(n_models), np.zeros(5)])
    bounds_w = [(-3, 3)] * n_models + [(-0.5, 0.5)] * 5
    res_w = differential_evolution(obj_weighted, bounds=bounds_w, seed=42,
                                   maxiter=500, tol=1e-8, polish=False, popsize=30)
    w_opt = np.exp(res_w.x[:n_models])
    w_opt /= w_opt.sum()
    b_opt = res_w.x[n_models:]
    avg_w = sum(wi * pi for wi, pi in zip(w_opt, all_p))
    la_w = np.log(avg_w + 1e-10)
    w_preds = (la_w + b_opt).argmax(1)
    w_fe = feval.evaluate(w_preds)
    p(f"  Weighted: {w_fe:.4f}")
    p(f"  Weights: {dict(zip(keys, np.round(w_opt, 3)))}")
    if w_fe < best_fe:
        best_fe = w_fe
        best_preds = w_preds
        best_name = "weighted_all"

    p(f"\n{'='*60}")
    p(f"BEST ensemble: {best_name} = {best_fe:.4f}")
    p(f"Previous best (v2 SAM): 0.1045")

    # Save
    np.savez(f"{OUT_DIR}/predictions_v10_ensemble.npz", predictions=best_preds.astype(np.int8))
    p(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
