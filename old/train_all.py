"""Train all key models on the unified pipeline and evaluate through verify.py.

Loads from data/composition_{train,test}/ (standardized .npy files).
No DE optimization — raw fraction error only.
Trains on FULL training set (no internal val split).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import time, subprocess, os, gc

DEVICE = "cuda"
BATCH_SIZE = 2048
SEED = 42
OUT_DIR = "submissions/unified"

def p(msg): print(msg, flush=True)

# ============================================================
# Data
# ============================================================

def load_data():
    p("Loading standardized composition data...")
    trn_mat = np.array(np.load("data/composition_train/matrices.npy", mmap_mode="r"))
    trn_feat = np.array(np.load("data/composition_train/features.npy", mmap_mode="r"))
    trn_labels = np.array(np.load("data/composition_train/labels_composition.npy", mmap_mode="r"))
    tst_mat = np.array(np.load("data/composition_test/matrices.npy", mmap_mode="r"))
    tst_feat = np.array(np.load("data/composition_test/features.npy", mmap_mode="r"))
    tst_labels = np.array(np.load("data/composition_test/labels_composition.npy", mmap_mode="r"))
    p(f"  Train: {len(trn_labels):,}  Test: {len(tst_labels):,}")
    return trn_mat, trn_feat, trn_labels, tst_mat, tst_feat, tst_labels


def make_reco_4(feat):
    """4 features for LeNet: Ne, Nmu, Age, Ze (published normalization)."""
    Ne, Nmu, Age, Ze = feat[:, 3], feat[:, 4], feat[:, 5], feat[:, 1]
    return torch.from_numpy(np.column_stack([
        (Ne-5.31)/0.5, (Nmu-4.3)/0.42, Age-1.0, Ze/60.0
    ]).astype(np.float32))


def make_reco_6(feat):
    """6 features for HybridModel: Ne, Nmu, Age, Ze, Ne-Nmu, E."""
    E, Ze, Ne, Nmu, Age = feat[:,0], feat[:,1], feat[:,3], feat[:,4], feat[:,5]
    return torch.from_numpy(np.column_stack([
        (Ne-5.31)/0.5, (Nmu-4.3)/0.42, Age-1.0, Ze/60.0,
        (Ne-Nmu-0.8)/0.3, (E-15.5)/1.0
    ]).astype(np.float32))


class DS(Dataset):
    def __init__(self, mat, reco, labels, augment=False, log1p=False, extra_aug=False):
        self.augment = augment
        self.extra_aug = extra_aug  # cutout + noise (only for HybridModel)
        m = torch.from_numpy(mat.astype(np.float32))
        self.x = torch.log1p(m) if log1p else m
        self.reco = reco
        self.y = torch.from_numpy(labels.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        a = self.x[idx]
        if self.augment:
            # Base augmentation (matches published kgnn: Rotate90 + Flip)
            if torch.rand(1)<0.5: a = torch.rot90(a, 1, [0,1])
            if torch.rand(1)<0.5: a = torch.rot90(a, 2, [0,1])
            if torch.rand(1)<0.5: a = torch.flip(a, dims=[0])
            if self.extra_aug:
                # Additional augmentation for HybridModel only
                if torch.rand(1)<0.3:
                    cx,cy = torch.randint(0,13,(1,)).item(), torch.randint(0,13,(1,)).item()
                    a = a.clone(); a[cy:cy+4,cx:cx+4,:] = 0
                if torch.rand(1)<0.3:
                    a = a + torch.randn_like(a) * 0.1 * a.std()
        return a, self.reco[idx], self.y[idx]


# ============================================================
# Architectures
# ============================================================

class LeNetArchitecture(nn.Module):
    def __init__(self, n_classes=5, n_reco_features=4, n_channels=2):
        super().__init__()
        ch = 16
        self.conv0 = nn.Conv2d(n_channels, ch, 2)
        self.conv1 = nn.Conv2d(ch, ch, 3)
        self.conv2 = nn.Conv2d(ch, ch, 4)
        self.conv3 = nn.Conv2d(ch, ch, 2)
        self.fc1 = nn.Linear(ch*3*3 + n_reco_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.fc_on_conv = nn.Linear(ch*3*3, n_classes)
    def forward(self, x, x_reco=None, return_cnn_out=False):
        x = x.permute(0,3,1,2).float()
        x = F.max_pool2d(F.relu(self.conv0(x)), 2, stride=1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x_cnn = x
        if x_reco is not None: x = torch.cat((x, x_reco), 1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = self.fc3(x)
        return (x, self.fc_on_conv(x_cnn)) if return_cnn_out else x


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch//r,8)), nn.ReLU(), nn.Linear(max(ch//r,8), ch), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class HybridModel(nn.Module):
    def __init__(self, n_feat=6, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256), nn.AdaptiveAvgPool2d(1))
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(
            nn.Linear(256+128,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,n_classes))
    def forward(self, mat, feat):
        mat = mat.permute(0,3,1,2)
        return self.head(torch.cat([self.cnn(mat).flatten(1), self.feat_mlp(feat)], 1))


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    @torch.no_grad()
    def first_step(self):
        gn = torch.stack([pp.grad.norm(2) for g in self.param_groups for pp in g["params"] if pp.grad is not None]).norm(2)
        for group in self.param_groups:
            scale = group["rho"] / (gn + 1e-12)
            for pp in group["params"]:
                if pp.grad is None: continue
                e_w = pp.grad * scale; pp.add_(e_w); self.state[pp]["e_w"] = e_w
    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for pp in group["params"]:
                if pp.grad is None: continue
                pp.sub_(self.state[pp]["e_w"])
        self.base_optimizer.step()
    def zero_grad(self): self.base_optimizer.zero_grad()


# ============================================================
# Loss helpers
# ============================================================

def lenet_loss(model, a, r, y, ce):
    """LeNet loss: CE/5 + CNN-only CE/5 + batch fraction MSE."""
    out, out_cnn = model(a, r, return_cnn_out=True)
    loss = ce(out.float(), y).mean() / 5 + ce(out_cnn.float(), y).mean() / 5
    # batch fraction loss
    pr = F.softmax(out.float(), 1)
    ip,cp = torch.unique(pr.argmax(1), sorted=True, return_counts=True)
    yp = torch.zeros(5, device=a.device); yp[ip.long()] = cp.float()
    it,ct = torch.unique(y, sorted=True, return_counts=True)
    yt = torch.zeros(5, device=a.device); yt[it.long()] = ct.float()
    loss += ((yp/yp.sum() - yt/yt.sum())**2).mean()
    return loss, out


def hybrid_loss(model, a, r, y, ce):
    """HybridModel loss: CE + batch fraction MSE."""
    out = model(a, r)
    loss = ce(out.float(), y)
    pr = F.softmax(out.float(), 1)
    ip,cp = torch.unique(pr.argmax(1), sorted=True, return_counts=True)
    yp = torch.zeros(5, device=a.device); yp[ip.long()] = cp.float()
    it,ct = torch.unique(y, sorted=True, return_counts=True)
    yt = torch.zeros(5, device=a.device); yt[it.long()] = ct.float()
    loss = loss + ((yp/yp.sum() - yt/yt.sum())**2).mean()
    return loss, out


# ============================================================
# Training loop
# ============================================================

def predict_test(model, test_ds):
    model.eval()
    all_p = []
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for a, r, y in loader:
            a, r = a.to(DEVICE), r.to(DEVICE)
            out = model(a, r) if not hasattr(model, 'fc_on_conv') else model(a, r)
            all_p.append(F.softmax(out.float(), 1).cpu().numpy())
    return np.concatenate(all_p)


def run_verify(pred_path, name):
    result = subprocess.run(["uv", "run", "python", "verify.py", pred_path],
                            capture_output=True, text=True, timeout=600)
    for line in result.stdout.split('\n'):
        if any(k in line for k in ['KEY METRIC', 'ACCURACY', 'MeanAE', 'MaxAE', 'P99AE', 'P95AE']):
            p(f"  [{name}] {line.strip()}")


def train_experiment(name, model, train_ds, val_ds, test_ds, loss_fn, optimizer, scheduler,
                     max_epochs=80, use_sam=False, batch_size=None, patience_limit=15):
    """Train one experiment with val-based checkpoint selection, then evaluate on test.

    Checkpoint is saved based on val loss (no test data leakage).
    Early stopping based on val loss with configurable patience.
    Evaluates on test only at the end using best checkpoint.
    """
    t0 = time.time()
    bs = batch_size or BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"  [{name}] Params: {n_params:,}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    best_val_loss = float('inf')
    patience_counter = 0
    model_path = f"{OUT_DIR}/model_{name}.pt"

    for epoch in range(max_epochs):
        # Train
        model.train()
        tc = 0; tt = 0
        for a, r, y in train_loader:
            a, r, y = a.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
            if use_sam:
                loss, out = loss_fn(model, a, r, y)
                loss.backward()
                optimizer.first_step()
                optimizer.zero_grad()
                loss2, _ = loss_fn(model, a, r, y)
                loss2.backward()
                optimizer.second_step()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss, out = loss_fn(model, a, r, y)
                loss.backward()
                optimizer.step()
            tc += (out.argmax(1)==y).sum().item(); tt += len(y)

        # Validate every epoch
        model.eval()
        vl = 0; vt = 0
        with torch.no_grad():
            for a, r, y in val_loader:
                a, r, y = a.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
                loss, _ = loss_fn(model, a, r, y)
                vl += loss.item() * len(y); vt += len(y)
        val_loss = vl / vt

        # Scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else \
                 optimizer.base_optimizer.param_groups[0]['lr']
            p(f"  [{name}] Ep {epoch+1}/{max_epochs}: acc={tc/tt:.4f} val_loss={val_loss:.4f} lr={lr:.2e} [{time.time()-t0:.0f}s]")

        # Checkpoint + early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                p(f"  [{name}] Early stopping at epoch {epoch+1}")
                break

    # Load best checkpoint and evaluate on TEST
    model.load_state_dict(torch.load(model_path, weights_only=True))
    probs = predict_test(model, test_ds)
    preds = probs.argmax(1)
    from eval_utils import FractionErrorEvaluator
    fe = FractionErrorEvaluator(test_ds.y.numpy()).evaluate(preds)

    pred_path = f"{OUT_DIR}/predictions_{name}.npz"
    np.savez(pred_path, predictions=preds)
    p(f"  [{name}] BEST: {fe:.4f} ({(time.time()-t0)/60:.1f} min)")
    run_verify(pred_path, name)
    return fe


# ============================================================
# Main: all experiments
# ============================================================

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    trn_mat, trn_feat, trn_lab, tst_mat, tst_feat, tst_lab = load_data()
    reco4_trn, reco4_tst = make_reco_4(trn_feat), make_reco_4(tst_feat)
    reco6_trn, reco6_tst = make_reco_6(trn_feat), make_reco_6(tst_feat)

    _, counts = np.unique(trn_lab, return_counts=True)
    w = torch.tensor(counts[0]/counts, dtype=torch.float32)
    p(f"Class weights: {w.tolist()}")

    results = {}

    # Helper to create loss functions
    def ce_none(): return nn.CrossEntropyLoss(weight=w.to(DEVICE), reduction='none')
    def ce_mean(): return nn.CrossEntropyLoss(weight=w.to(DEVICE), label_smoothing=0.05)
    def ce_focal(gamma=2.0):
        ce_base = nn.CrossEntropyLoss(weight=w.to(DEVICE), reduction='none')
        def focal_loss(model, a, r, y):
            out = model(a, r)
            ce_val = ce_base(out.float(), y)
            pt = torch.exp(-ce_val)
            loss = ((1 - pt)**gamma * ce_val).mean()
            return loss, out
        return focal_loss

    def run_lenet():
        """Run reproduce_sota.py and then evaluate its model on composition_test.

        reproduce_sota.py loads raw QGS data, applies cuts, splits 70/21/9
        with val-based checkpoint selection — matching the published pipeline exactly.
        We then take the saved model and predict on our standardized test set.
        """
        torch.manual_seed(SEED); np.random.seed(SEED)
        t0 = time.time()
        name = "01_lenet"

        # Run reproduce_sota.py
        p(f"  [{name}] Running reproduce_sota.py...")
        result = subprocess.run(
            ["uv", "run", "python",
             "submissions/opus-composition-mar14/matched_pipeline/reproduce_sota.py"],
            capture_output=True, text=True, timeout=3600,
            env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                 "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "1")})
        for line in result.stdout.split('\n'):
            if 'fraction_error' in line or 'accuracy' in line:
                p(f"  [{name}] {line.strip()}")

        # Load saved model and predict on standardized test set
        model = LeNetArchitecture(n_reco_features=4).to(DEVICE)
        model.load_state_dict(torch.load("reproduce_sota_best.pt", weights_only=True))
        model.eval()

        test_ds = DS(tst_mat, reco4_tst, tst_lab)
        probs = predict_test(model, test_ds)
        preds = probs.argmax(1)
        from eval_utils import FractionErrorEvaluator
        fe = FractionErrorEvaluator(test_ds.y.numpy()).evaluate(preds)

        pred_path = f"{OUT_DIR}/predictions_{name}.npz"
        np.savez(pred_path, predictions=preds)
        torch.save(model.state_dict(), f"{OUT_DIR}/model_{name}.pt")
        p(f"  [{name}] BEST: {fe:.4f} ({(time.time()-t0)/60:.1f} min)")
        run_verify(pred_path, name)
        return fe

    # Fixed train/val split for all hybrid experiments (23% of train → val)
    n_trn = len(trn_lab)
    n_val = int(n_trn * 0.23)
    val_perm = torch.randperm(n_trn, generator=torch.Generator().manual_seed(SEED)).numpy()
    vi, ti = val_perm[:n_val], val_perm[n_val:]
    p(f"Hybrid train/val split: train={len(ti)}, val={len(vi)}")

    def run_hybrid(name, seed=42, use_sam=False, use_log1p=True,
                   label_smoothing=0.05, lr=1e-3, focal_gamma=None):
        torch.manual_seed(seed); np.random.seed(seed)
        model = HybridModel(n_feat=6).to(DEVICE)
        if focal_gamma is not None:
            loss_fn = ce_focal(focal_gamma)
        else:
            ce = nn.CrossEntropyLoss(weight=w.to(DEVICE), label_smoothing=label_smoothing)
            def loss_fn(mdl, a, r, y):
                return hybrid_loss(mdl, a, r, y, ce)

        if use_sam:
            opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer, T_max=80)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

        return train_experiment(name, model,
            DS(trn_mat[ti], reco6_trn[ti], trn_lab[ti], augment=True, log1p=use_log1p, extra_aug=True),
            DS(trn_mat[vi], reco6_trn[vi], trn_lab[vi], log1p=use_log1p),
            DS(tst_mat, reco6_tst, tst_lab, log1p=use_log1p),
            loss_fn, opt, sched, max_epochs=80, use_sam=use_sam)

    # ── Run all experiments ──

    p("\n" + "="*60 + "\n  1. LeNet (published baseline)")
    results['01_lenet'] = run_lenet()

    p("\n" + "="*60 + "\n  2. CNN+Attn seed 42")
    results['02_hybrid_s42'] = run_hybrid("02_hybrid_s42", seed=42)

    p("\n" + "="*60 + "\n  3. CNN+Attn seed 7")
    results['03_hybrid_s7'] = run_hybrid("03_hybrid_s7", seed=7)

    p("\n" + "="*60 + "\n  4. CNN+Attn seed 2026")
    results['04_hybrid_s2026'] = run_hybrid("04_hybrid_s2026", seed=2026)

    p("\n" + "="*60 + "\n  5. CNN+Attn no log1p")
    results['05_hybrid_nolog'] = run_hybrid("05_hybrid_nolog", use_log1p=False)

    p("\n" + "="*60 + "\n  6. CNN+Attn + SAM seed 42")
    results['06_sam_s42'] = run_hybrid("06_sam_s42", seed=42, use_sam=True)

    p("\n" + "="*60 + "\n  7. CNN+Attn + SAM seed 7")
    results['07_sam_s7'] = run_hybrid("07_sam_s7", seed=7, use_sam=True)

    p("\n" + "="*60 + "\n  8. CNN+Attn + SAM seed 2026")
    results['08_sam_s2026'] = run_hybrid("08_sam_s2026", seed=2026, use_sam=True)

    p("\n" + "="*60 + "\n  9. CNN+Attn + Focal loss (gamma=1)")
    results['09_focal_g1'] = run_hybrid("09_focal_g1", focal_gamma=1.0)

    p("\n" + "="*60 + "\n  10. CNN+Attn + Focal loss (gamma=2)")
    results['10_focal_g2'] = run_hybrid("10_focal_g2", focal_gamma=2.0)

    p("\n" + "="*60 + "\n  11. CNN+Attn no label smoothing")
    results['11_no_smooth'] = run_hybrid("11_no_smooth", label_smoothing=0.0)

    p("\n" + "="*60 + "\n  12. CNN+Attn lr=3e-4")
    results['12_lr3e4'] = run_hybrid("12_lr3e4", lr=3e-4)

    p("\n" + "="*60 + "\n  13. CNN+Attn + SAM lr=3e-4")
    results['13_sam_lr3e4'] = run_hybrid("13_sam_lr3e4", lr=3e-4, use_sam=True)

    # ── Summary ──
    p("\n" + "="*60)
    p("SUMMARY (raw fraction error, no DE):")
    p(f"{'#':<4} {'Model':<25} {'FracErr':>10}")
    p("-" * 44)
    for name, fe in sorted(results.items()):
        p(f"{name[:2]:<4} {name[3:]:<25} {fe:>10.4f}")

    best_name = min(results, key=results.get)
    p(f"\nBest: {best_name} = {results[best_name]:.4f}")
    p(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
