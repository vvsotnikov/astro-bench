"""Metric-aware stacking: a meta-learner trained with batch-fraction-matching loss
on top of per-agent softmaxes, vs. the per-event log-likelihood stackers tested in
stacking_experiment.py.

For each random 50/50 stratified split (5 seeds):
  - Concatenate Opus/GPT/Sonnet softmaxes (15-dim per event)
  - Train a small MLP on the stratified train half
  - Loss: CE + alpha * batch_fraction_MSE, sweep over alpha
  - Also try a "verifier-style" training scheme: each batch is a random mixture
    drawn from the train half, MSE on predicted vs target class fractions
  - Evaluate fraction error on the stratified eval half

Output: comparison table + JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "composition"))
from verify import _compute_fraction_error, _load_test_data  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/home/vladimir/cursor_projects/astro-bench-experiments")


def load_probs():
    opus = np.load(ROOT / "composition-opus-2apr/opus_v34_probs.npy").astype(np.float32)
    gpt = np.load(ROOT / "composition-gpt-2apr/gpt_v47_probs.npy").astype(np.float32)
    son_sum = np.load(ROOT / "composition-sonnet-27mar/logits_v44.npy").astype(np.float32)
    sonnet = son_sum / son_sum.sum(axis=1, keepdims=True)
    return opus, gpt, sonnet


class MetaMLP(nn.Module):
    def __init__(self, in_dim=15, hidden=64, n_classes=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def soft_batch_fraction_loss(logits, targets, n_classes=5):
    """Differentiable MSE between predicted soft fractions and true (one-hot) fractions."""
    probs = F.softmax(logits, dim=1)
    pred_frac = probs.mean(dim=0)
    true_frac = F.one_hot(targets, num_classes=n_classes).float().mean(dim=0)
    return F.mse_loss(pred_frac, true_frac)


def fraction_error_via_verify(preds, labels, eval_idx):
    return _compute_fraction_error(labels[eval_idx], preds.astype(int))["mean"]


def train_meta(X_tr, y_tr, X_ev, y_ev, *, alpha=1.0, epochs=200, lr=2e-3,
               batch_size=4096, seed=0, hidden=64, weight_decay=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.from_numpy(X_tr).float().to(DEVICE)
    yt = torch.from_numpy(y_tr).long().to(DEVICE)
    Xe = torch.from_numpy(X_ev).float().to(DEVICE)
    ye = torch.from_numpy(y_ev).long().to(DEVICE)

    model = MetaMLP(in_dim=Xt.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(Xt)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx])
            ce = F.cross_entropy(logits, yt[idx])
            bf = soft_batch_fraction_loss(logits, yt[idx])
            loss = ce + alpha * bf
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(Xe).argmax(dim=1).cpu().numpy()
    return preds


def train_verifier_style(X_tr, y_tr, X_ev, y_ev, *, n_grid_points=200,
                          mixture_size=2000, epochs=400, lr=2e-3, seed=0,
                          hidden=64, weight_decay=1e-4):
    """Train with batches that mimic the verifier's mixture-grid sampling.

    Each "batch" is a random mixture: draw a random fraction vector over 5 classes,
    sample mixture_size events from the train half with those fractions, compute
    MSE between predicted soft fractions and the target fraction vector.
    """
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(X_tr).float().to(DEVICE)
    yt_np = y_tr  # numpy
    Xe = torch.from_numpy(X_ev).float().to(DEVICE)

    # Build per-class indices in train set
    cls_idx = {c: np.where(yt_np == c)[0] for c in range(5)}

    model = MetaMLP(in_dim=Xt.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        for _ in range(n_grid_points):
            # Random fraction vector (Dirichlet, then renormalize)
            frac = rng.dirichlet(np.ones(5))
            counts = np.round(frac * mixture_size).astype(int)
            diff = mixture_size - counts.sum()
            if diff != 0:
                counts[np.argmax(counts)] += diff
            sample_idx = []
            for c in range(5):
                if counts[c] > 0 and len(cls_idx[c]) > 0:
                    sample_idx.append(rng.choice(cls_idx[c], size=counts[c], replace=True))
            sample_idx = np.concatenate(sample_idx)
            target_frac = torch.from_numpy(
                np.array([counts[c] / counts.sum() for c in range(5)], dtype=np.float32),
            ).to(DEVICE)
            X_batch = Xt[sample_idx]
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            pred_frac = probs.mean(dim=0)
            loss = F.mse_loss(pred_frac, target_frac)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(Xe).argmax(dim=1).cpu().numpy()
    return preds


def main():
    labels, _ = _load_test_data()
    opus, gpt, sonnet = load_probs()
    X = np.concatenate([opus, gpt, sonnet], axis=1)
    print(f"Test events: {len(labels)}, X shape: {X.shape}")

    # Reference: best single agent (Opus) on the eval half (computed per seed)
    opus_pred = opus.argmax(axis=1)

    seeds = [42, 1, 7, 100, 2026]
    configs = [
        ("MA-MLP CE+α=1",   {"alpha": 1.0,    "epochs": 200}),
        ("MA-MLP CE+α=10",  {"alpha": 10.0,   "epochs": 200}),
        ("MA-MLP CE+α=100", {"alpha": 100.0,  "epochs": 200}),
        ("MA-MLP CE+α=1000",{"alpha": 1000.0, "epochs": 300}),
    ]
    verifier_style = True

    rows = {name: [] for name, _ in configs}
    rows["Opus alone (eval)"] = []
    rows["MA-MLP verifier-style"] = []

    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        train_idx, eval_idx = next(sss.split(X, labels))
        X_tr, y_tr = X[train_idx], labels[train_idx]
        X_ev, y_ev = X[eval_idx], labels[eval_idx]

        opus_eval_fe = fraction_error_via_verify(opus_pred[eval_idx], labels, eval_idx)
        rows["Opus alone (eval)"].append(opus_eval_fe)

        for name, cfg in configs:
            preds = train_meta(X_tr, y_tr, X_ev, y_ev, seed=seed, **cfg)
            fe = fraction_error_via_verify(preds, labels, eval_idx)
            rows[name].append(fe)
            print(f"  seed {seed:>4d}  {name:>22s}  fe={fe:.5f}  vs Opus {opus_eval_fe:.5f}")

        if verifier_style:
            preds = train_verifier_style(X_tr, y_tr, X_ev, y_ev, seed=seed,
                                          n_grid_points=200, mixture_size=2000, epochs=200)
            fe = fraction_error_via_verify(preds, labels, eval_idx)
            rows["MA-MLP verifier-style"].append(fe)
            print(f"  seed {seed:>4d}  {'MA-MLP verifier-style':>22s}  fe={fe:.5f}")

    sigma = 4e-4  # noise floor
    opus_mean = np.mean(rows["Opus alone (eval)"])

    print("\n=== Summary ===")
    print(f"  {'method':>26s}  {'mean FE':>10s}  {'± std':>9s}  {'Δ vs Opus':>12s}")
    for name, vals in rows.items():
        a = np.array(vals)
        delta = a.mean() - opus_mean
        print(f"  {name:>26s}  {a.mean():.5f}  {a.std():.5f}  {delta:+.5f}  ({delta/sigma:+.2f}σ)")

    # Save
    out = {name: [float(v) for v in vals] for name, vals in rows.items()}
    out["_meta"] = {"sigma_noise": sigma, "opus_mean_eval_fe": float(opus_mean)}
    Path(__file__).resolve().parent.joinpath("metric_aware_stacker_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved metric_aware_stacker_results.json")


if __name__ == "__main__":
    main()
