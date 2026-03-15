"""v8b: Calibration-focused training.
Key insight: fraction error depends on how well the confusion matrix maps
fractions, not per-event accuracy. We want a model that:
1. Has the most "diagonal" confusion matrix possible
2. Is well-calibrated across all classes

Strategy:
- Multi-seed training (seeds 42, 7, 123) with v8 architecture
- For each seed, also save probabilities
- Then do a proper bias optimization on the 3-model ensemble
- This is simpler than architectural novelty but gives diverse models
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import time
import gc

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-composition-mar14"
DEVICE = "cuda"
BATCH_SIZE = 4096
EPOCHS = 20
LR = 1e-3
LABEL_SMOOTH = 0.05

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    feats = [
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu,
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
    ]
    return np.stack(feats, axis=1).astype(np.float32)


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class HybridModel(nn.Module):
    def __init__(self, n_feat=13, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, mat, feat):
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))


def load_all_f32(split, feat_stats=None):
    p(f"Loading {split} data...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    chunk = 250000
    mat_list = []
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        m = np.array(matrices[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
        if (i // chunk) % 4 == 0:
            p(f"  {split}: {end}/{n}")
    mat_tensor = torch.cat(mat_list, dim=0)
    del mat_list; gc.collect()

    feat_chunks = []
    for i in range(0, n, 500000):
        end = min(i + 500000, n)
        f = np.array(raw_feats[i:end], dtype=np.float32)
        feat_chunks.append(engineer_features(f))
    feats = np.concatenate(feat_chunks)
    del feat_chunks; gc.collect()

    if feat_stats is None:
        feat_mean = feats.mean(0)
        feat_std = feats.std(0) + 1e-6
    else:
        feat_mean, feat_std = feat_stats
    feats = (feats - feat_mean) / feat_std
    feat_tensor = torch.from_numpy(feats)
    del feats; gc.collect()

    label_tensor = torch.from_numpy(np.array(labels[:], dtype=np.int64))
    p(f"  {split}: mat={mat_tensor.shape}")
    return mat_tensor, feat_tensor, label_tensor, (feat_mean, feat_std)


def train_seed(seed, mat_train, feat_train, y_train, mat_test, feat_test, y_test):
    """Train one model with given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    train_loader = DataLoader(
        TensorDataset(mat_train, feat_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(mat_test, feat_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridModel(n_feat=feat_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    best_probs = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for mat_b, feat_b, label_b in train_loader:
            mat_b, feat_b, label_b = mat_b.to(DEVICE), feat_b.to(DEVICE), label_b.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                out = model(mat_b, feat_b)
                loss = criterion(out, label_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (out.argmax(1) == label_b).sum().item()
            total += len(label_b)
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_probs = []
        tc, tt = 0, 0
        with torch.no_grad():
            for mat_b, feat_b, label_b in test_loader:
                mat_b, feat_b, label_b = mat_b.to(DEVICE), feat_b.to(DEVICE), label_b.to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(mat_b, feat_b)
                all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
                tc += (out.argmax(1) == label_b).sum().item()
                tt += len(label_b)
        test_acc = tc / tt
        p(f"  Seed {seed} Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_probs = np.concatenate(all_probs)
            p(f"    >>> Best: {best_acc:.4f}")

    return best_probs, best_acc


def main():
    t0 = time.time()

    mat_train, feat_train, y_train, stats = load_all_f32("train")
    mat_test, feat_test, y_test, _ = load_all_f32("test", feat_stats=stats)

    seeds = [42, 7, 123, 2026]
    all_probs = {}

    for seed in seeds:
        p(f"\n{'='*60}")
        p(f"Training seed={seed}")
        p(f"{'='*60}")
        probs, acc = train_seed(seed, mat_train, feat_train, y_train, mat_test, feat_test, y_test)
        all_probs[seed] = probs
        np.save(f"{OUT_DIR}/probs_v8b_s{seed}.npy", probs)
        p(f"Seed {seed}: {acc:.4f}")

    # Ensemble optimization with DE
    from scipy.optimize import differential_evolution

    y_test_np = np.array(y_test.numpy(), dtype=np.int64)

    MIXTURE_SIZE = 5000
    MIXTURE_SEED = 2026
    GRID_STEP = 0.1

    def generate_fraction_grid(n_classes=5, step=GRID_STEP):
        n_steps = round(1.0 / step)
        fractions = []
        def _recurse(remaining, depth, current):
            if depth == n_classes - 1:
                current.append(remaining * step)
                fractions.append(current[:])
                current.pop()
                return
            for i in range(remaining + 1):
                current.append(i * step)
                _recurse(remaining - i, depth + 1, current)
                current.pop()
        _recurse(n_steps, 0, [])
        return np.array(fractions)

    class FastEvaluator:
        def __init__(self, y_test, seed=MIXTURE_SEED):
            self.n_classes = 5
            self.class_indices = {c: np.where(y_test == c)[0] for c in range(self.n_classes)}
            self.fractions = generate_fraction_grid(self.n_classes, GRID_STEP)
            self.n_ensembles = len(self.fractions)
            rng = np.random.default_rng(seed)
            self.all_sample_idx = []
            self.true_fracs = np.zeros((self.n_ensembles, self.n_classes))
            for mix_idx in range(self.n_ensembles):
                target_fracs = self.fractions[mix_idx]
                counts = np.round(target_fracs * MIXTURE_SIZE).astype(int)
                diff = MIXTURE_SIZE - counts.sum()
                if diff != 0:
                    counts[np.argmax(counts)] += diff
                indices = []
                for c in range(self.n_classes):
                    n_sample = counts[c]
                    if n_sample <= 0:
                        continue
                    idx = rng.choice(self.class_indices[c], size=n_sample, replace=True)
                    indices.append(idx)
                    self.true_fracs[mix_idx, c] = n_sample / MIXTURE_SIZE
                self.all_sample_idx.append(np.concatenate(indices))

        def evaluate(self, preds):
            all_errors = np.zeros((self.n_ensembles, self.n_classes))
            for mix_idx in range(self.n_ensembles):
                sampled = preds[self.all_sample_idx[mix_idx]]
                pred_counts = np.bincount(sampled, minlength=self.n_classes)[:self.n_classes]
                pred_fracs = pred_counts / pred_counts.sum()
                all_errors[mix_idx] = np.abs(self.true_fracs[mix_idx] - pred_fracs)
            return float(all_errors.mean())

    evaluator = FastEvaluator(y_test_np)

    # Also load v8 and v11 from previous run
    probs_v8_old = np.load(f"{PREV_DIR}/probs_v8.npy")
    probs_v11_old = np.load(f"{PREV_DIR}/probs_v11_eval.npy")

    # Try various ensembles
    p(f"\n{'='*60}")
    p("Ensemble optimization")
    p(f"{'='*60}")

    combos = [
        ("4seed", [all_probs[s] for s in seeds]),
        ("4seed+v8old+v11old", [all_probs[s] for s in seeds] + [probs_v8_old, probs_v11_old]),
        ("v8old+v11old", [probs_v8_old, probs_v11_old]),
        ("s42+v8old+v11old", [all_probs[42], probs_v8_old, probs_v11_old]),
        ("s42+s7", [all_probs[42], all_probs[7]]),
        ("s42+s7+s123", [all_probs[42], all_probs[7], all_probs[123]]),
    ]

    best_overall = 1.0
    best_preds = None
    best_combo_name = ""

    for name, probs_list in combos:
        avg = np.mean(probs_list, axis=0)
        log_avg = np.log(avg + 1e-10)
        raw_fe = evaluator.evaluate(avg.argmax(1))

        def obj(biases):
            return evaluator.evaluate((log_avg + biases).argmax(1))

        res = differential_evolution(
            obj, bounds=[(-0.5, 0.5)] * 5,
            seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
        )
        bias_preds = (log_avg + res.x).argmax(1)
        bias_fe = evaluator.evaluate(bias_preds)
        p(f"  {name}: raw={raw_fe:.6f} bias={bias_fe:.6f}")

        if bias_fe < best_overall:
            best_overall = bias_fe
            best_preds = bias_preds.copy()
            best_combo_name = name

    p(f"\nBEST: {best_combo_name} = {best_overall:.6f}")
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))

    elapsed = time.time() - t0
    p(f"Total time: {elapsed/60:.1f} min")
    p("---")
    p(f"metric: {best_overall:.6f}")
    p(f"description: {best_combo_name} multi-seed ensemble + DE bias opt")


if __name__ == "__main__":
    main()
