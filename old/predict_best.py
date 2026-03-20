"""Generate verify.py-compatible predictions from the best CNN+Attn+MLP (SAM) model.
Saves both argmax predictions and softmax probabilities for DE optimization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eval_utils import FractionErrorEvaluator, generate_fraction_grid
from scipy.optimize import differential_evolution
import time

DEVICE = "cuda"
MODEL_PATH = "submissions/opus-composition-mar14/matched_pipeline/model_v2_sam.pt"
MIXTURE_SIZE = 5000
MIXTURE_SEED = 2026

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch//r, 8)), nn.ReLU(),
            nn.Linear(max(ch//r, 8), ch), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class HybridModel(nn.Module):
    def __init__(self, n_feat=6, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256), nn.AdaptiveAvgPool2d(1))
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(
            nn.Linear(256+128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes))
    def forward(self, mat, feat):
        mat = mat.permute(0, 3, 1, 2)
        return self.head(torch.cat([self.cnn(mat).flatten(1), self.feat_mlp(feat)], dim=1))


class RobustEvaluator:
    """Evaluator that can optimize for MeanAE, MaxAE, or P99AE."""

    def __init__(self, y_test, seed=MIXTURE_SEED):
        self.n_classes = 5
        class_indices = {c: np.where(y_test == c)[0] for c in range(self.n_classes)}
        self.fractions = generate_fraction_grid()
        self.n_ensembles = len(self.fractions)
        rng = np.random.default_rng(seed)

        self.sample_indices = []
        self.true_fracs = np.zeros((self.n_ensembles, self.n_classes))

        for mix_idx in range(self.n_ensembles):
            counts = np.round(self.fractions[mix_idx] * MIXTURE_SIZE).astype(int)
            diff = MIXTURE_SIZE - counts.sum()
            if diff != 0:
                counts[np.argmax(counts)] += diff
            indices = []
            for c in range(self.n_classes):
                if counts[c] <= 0:
                    continue
                idx = rng.choice(class_indices[c], size=counts[c], replace=True)
                indices.append(idx)
                self.true_fracs[mix_idx, c] = counts[c] / MIXTURE_SIZE
            self.sample_indices.append(np.concatenate(indices))

    def evaluate_full(self, preds):
        """Returns (n_ensembles, n_classes) error matrix."""
        errors = np.zeros((self.n_ensembles, self.n_classes))
        for i in range(self.n_ensembles):
            sampled = preds[self.sample_indices[i]]
            pred_counts = np.bincount(sampled, minlength=self.n_classes)[:self.n_classes]
            pred_fracs = pred_counts / pred_counts.sum()
            errors[i] = np.abs(self.true_fracs[i] - pred_fracs)
        return errors

    def mean_ae(self, preds):
        return float(self.evaluate_full(preds).mean())

    def max_ae(self, preds):
        """Max across per-class max errors (worst-case robustness)."""
        errors = self.evaluate_full(preds)
        return float(errors.max(axis=0).mean())  # mean of per-class max

    def p99_ae(self, preds):
        errors = self.evaluate_full(preds)
        return float(np.percentile(errors, 99, axis=0).mean())


def run_de(log_probs, evaluator, objective="mean", label=""):
    """Run DE optimization for a given objective."""
    def obj_mean(b):
        return evaluator.mean_ae((log_probs + b).argmax(1))
    def obj_max(b):
        return evaluator.max_ae((log_probs + b).argmax(1))
    def obj_p99(b):
        return evaluator.p99_ae((log_probs + b).argmax(1))

    obj_fn = {"mean": obj_mean, "max": obj_max, "p99": obj_p99}[objective]

    t0 = time.time()
    res = differential_evolution(
        obj_fn,
        bounds=[(-0.5, 0.5)] * 5,
        seed=42,
        maxiter=500,
        tol=1e-8,
        polish=False,
        popsize=25,
    )
    elapsed = time.time() - t0
    preds = (log_probs + res.x).argmax(1)

    errors = evaluator.evaluate_full(preds)
    mean_ae = float(errors.mean())
    per_class_max = errors.max(axis=0)
    per_class_mean = errors.mean(axis=0)
    per_class_p99 = np.percentile(errors, 99, axis=0)

    names = ['pr', 'he', 'ca', 'si', 'ir']
    print(f"\n  DE ({label}, obj={objective}) [{elapsed:.0f}s]:", flush=True)
    print(f"    MeanAE: {mean_ae:.4f}  |  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, per_class_mean)), flush=True)
    print(f"    MaxAE:  {per_class_max.mean():.4f}  |  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, per_class_max)), flush=True)
    print(f"    P99AE:  {per_class_p99.mean():.4f}  |  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, per_class_p99)), flush=True)
    print(f"    biases: {np.round(res.x, 4).tolist()}", flush=True)

    return preds, res.x, mean_ae


def main():
    model = HybridModel(n_feat=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    matrices = np.load("data/composition_test/matrices.npy", mmap_mode="r")
    features = np.load("data/composition_test/features.npy", mmap_mode="r")
    labels = np.load("data/composition_test/labels_composition.npy")
    n = len(matrices)
    print(f"Test set: {n} events", flush=True)

    # Generate probabilities
    all_probs = []
    BS = 2048
    with torch.no_grad():
        for i in range(0, n, BS):
            mat = torch.log1p(torch.from_numpy(np.array(matrices[i:i+BS]).astype(np.float32))).to(DEVICE)
            feat = np.array(features[i:i+BS]).astype(np.float32)
            E, Ze, Ne, Nmu, Age = feat[:, 0], feat[:, 1], feat[:, 3], feat[:, 4], feat[:, 5]
            reco = torch.from_numpy(np.column_stack([
                (Ne - 5.31) / 0.5, (Nmu - 4.3) / 0.42, Age - 1.0, Ze / 60.0,
                (Ne - Nmu - 0.8) / 0.3, (E - 15.5) / 1.0,
            ]).astype(np.float32)).to(DEVICE)
            out = model(mat, reco)
            all_probs.append(F.softmax(out.float(), dim=1).cpu().numpy())

    probs = np.concatenate(all_probs)
    log_probs = np.log(probs + 1e-10)

    evaluator = RobustEvaluator(labels)

    # Raw (no DE)
    raw_preds = probs.argmax(1)
    errors = evaluator.evaluate_full(raw_preds)
    names = ['pr', 'he', 'ca', 'si', 'ir']
    print(f"\n  Raw (no DE):", flush=True)
    print(f"    MeanAE: {errors.mean():.4f}  |  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, errors.mean(axis=0))), flush=True)
    print(f"    MaxAE:  {errors.max(axis=0).mean():.4f}  |  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, errors.max(axis=0))), flush=True)
    print(f"    P99AE:  {np.percentile(errors, 99, axis=0).mean():.4f}  |  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, np.percentile(errors, 99, axis=0))), flush=True)

    # DE optimizing MeanAE
    preds_mean, biases_mean, fe_mean = run_de(log_probs, evaluator, "mean", "best model")

    # DE optimizing MaxAE
    preds_max, biases_max, fe_max_mean = run_de(log_probs, evaluator, "max", "best model")

    # DE optimizing P99AE
    preds_p99, biases_p99, fe_p99_mean = run_de(log_probs, evaluator, "p99", "best model")

    # Save the MeanAE-optimized predictions for verify.py
    np.savez("predictions_best_de_mean.npz", predictions=preds_mean)
    np.savez("predictions_best_de_max.npz", predictions=preds_max)
    np.savez("predictions_best_de_p99.npz", predictions=preds_p99)
    print(f"\nSaved all prediction files", flush=True)


if __name__ == "__main__":
    main()
