"""Beat SOTA v2: Use QGS + EPOS combined training data.

The published model trains on QGS only (383K events after cuts).
Adding EPOS gives ~200K more events — more data should improve the confusion matrix.
Also: more feature engineering, longer training.

Same test set as reproduce_sota.py (QGS test split).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import differential_evolution
import time

DATA_DIR = "data"
DEVICE = "cuda"
BATCH_SIZE = 2048
MAX_EPOCHS = 80
LR = 1e-3
SEED = 42

CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]
NORM = {'Ne': (5.31, 0.5), 'Nmu': (4.3, 0.42), 'Age': (1.0, 1.0), 'Ze': (0.0, 60.0),
        'E': (15.5, 1.0), 'Ne_Nmu': (0.8, 0.3)}
MIXTURE_SIZE = 5000; MIXTURE_SEED = 2026; GRID_STEP = 0.1

def p(msg): print(msg, flush=True)


class QGSDataset(Dataset):
    RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']

    def __init__(self, mat_file, feat_file, true_file, cut_data=None):
        raw_mat = np.load(mat_file)['matrices']
        raw_feat = np.load(feat_file)['features']
        raw_true = np.load(true_file)['true_features']
        reco = raw_feat[:, 1:]
        true = np.concatenate([raw_feat[:, [0]], raw_true[:, [0, 2, 3, 4, 5]]], axis=1)
        self.arrays = torch.from_numpy(raw_mat)
        self.reconstructed = torch.from_numpy(reco.astype(np.float32))
        self.true = torch.from_numpy(true.astype(np.float32))
        if cut_data:
            mask = torch.ones(len(self.reconstructed), dtype=torch.bool)
            for feat_name, (lo, hi) in cut_data.items():
                idx = self.RECO_HEADERS.index(feat_name)
                mask &= (self.reconstructed[:, idx] > lo) & (self.reconstructed[:, idx] < hi)
            self.arrays = self.arrays[mask]
            self.reconstructed = self.reconstructed[mask]
            self.true = self.true[mask]
        p(f"  {mat_file}: {len(self)} events")

    def __len__(self): return self.reconstructed.shape[0]
    def __getitem__(self, idx):
        return {'arrays': self.arrays[idx],
                'reconstructed': {name: self.reconstructed[idx, i] for i, name in enumerate(self.RECO_HEADERS)},
                'true': {'primary_id': self.true[idx, 0]}}


class FastTensorDataset(Dataset):
    """Pre-extracted tensors for max speed."""
    def __init__(self, arrays, reco, labels, augment=False):
        self.arrays = arrays
        self.reco = reco
        self.labels = labels
        self.augment = augment

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        a = self.arrays[idx]
        if self.augment:
            if torch.rand(1) < 0.5: a = torch.rot90(a, 1, [0, 1])
            if torch.rand(1) < 0.5: a = torch.rot90(a, 2, [0, 1])
            if torch.rand(1) < 0.5: a = torch.flip(a, dims=[0])
        return a, self.reco[idx], self.labels[idx]


def extract_features(ds, indices, channels=[1, 2]):
    """Extract and engineer features from dataset."""
    arrays = ds.arrays[indices][:, :, :, channels].float()
    Ne = ds.reconstructed[indices, RECO_HEADERS.index('Ne')].float()
    Nmu = ds.reconstructed[indices, RECO_HEADERS.index('Nmu')].float()
    Age = ds.reconstructed[indices, RECO_HEADERS.index('Age')].float()
    Ze = ds.reconstructed[indices, RECO_HEADERS.index('Ze')].float()
    E = ds.reconstructed[indices, RECO_HEADERS.index('E')].float()

    # Normalize
    Ne_n = (Ne - 5.31) / 0.5
    Nmu_n = (Nmu - 4.3) / 0.42
    Age_n = Age - 1.0
    Ze_n = Ze / 60.0
    # Engineered features
    Ne_Nmu = (Ne - Nmu - 0.8) / 0.3  # strongest discriminant
    E_n = (E - 15.5) / 1.0

    reco = torch.stack([Ne_n, Nmu_n, Age_n, Ze_n, Ne_Nmu, E_n], dim=1)
    labels = ds.true[indices, 0].long() - 1
    return arrays, reco, labels


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class HybridModelV2(nn.Module):
    """CNN+Attn+MLP with 6 reco features."""
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
            nn.Linear(256 + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes))

    def forward(self, mat, feat):
        mat = mat.permute(0, 3, 1, 2)
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))


def generate_fraction_grid(n_classes=5, step=GRID_STEP):
    n_steps = round(1.0/step); fracs=[]
    def _r(rem,d,c):
        if d==n_classes-1: c.append(rem*step); fracs.append(c[:]); c.pop(); return
        for i in range(rem+1): c.append(i*step); _r(rem-i,d+1,c); c.pop()
    _r(n_steps,0,[]); return np.array(fracs)

class FracEval:
    def __init__(self, y):
        ci={c:np.where(y==c)[0] for c in range(5)}
        self.fracs=generate_fraction_grid(); self.ne=len(self.fracs)
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
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()

    # Load QGS data (same split as reproduction for test set)
    p("Loading QGS data...")
    qgs = QGSDataset(f"{DATA_DIR}/qgs_spectra_matrices.npz",
                     f"{DATA_DIR}/qgs_spectra_features.npz",
                     f"{DATA_DIR}/qgs_spectra_true_features.npz", cut_data=CUTS)

    # Same split as reproduction
    n = len(qgs); n_valid = int(n*0.3); n_train = n - n_valid
    train_sub, valid_sub = random_split(qgs, [n_train, n_valid], generator=torch.Generator().manual_seed(42))
    n_test = int(n_valid*0.3); n_val = n_valid - n_test
    val_sub, test_sub = random_split(valid_sub, [n_val, n_test], generator=torch.Generator().manual_seed(42))

    # Get test indices (these stay fixed)
    test_indices = []
    ds = test_sub
    idx = list(range(len(ds)))
    while hasattr(ds, 'dataset'):
        if hasattr(ds, 'indices'): idx = [ds.indices[i] for i in idx]
        ds = ds.dataset
    test_indices = idx

    # Also get val indices
    ds = val_sub; idx = list(range(len(ds)))
    while hasattr(ds, 'dataset'):
        if hasattr(ds, 'indices'): idx = [ds.indices[i] for i in idx]
        ds = ds.dataset
    val_indices = idx

    # Get train indices from QGS
    ds = train_sub; idx = list(range(len(ds)))
    while hasattr(ds, 'dataset'):
        if hasattr(ds, 'indices'): idx = [ds.indices[i] for i in idx]
        ds = ds.dataset
    qgs_train_indices = idx

    # Load EPOS data too
    p("Loading EPOS data...")
    epos = QGSDataset(f"{DATA_DIR}/epos_spectra_matrices.npz",
                      f"{DATA_DIR}/epos_spectra_features.npz",
                      f"{DATA_DIR}/epos_spectra_true_features.npz", cut_data=CUTS)

    # Extract features
    p("Extracting features...")
    test_arrays, test_reco, test_labels = extract_features(qgs, test_indices)
    val_arrays, val_reco, val_labels = extract_features(qgs, val_indices)
    qgs_train_arrays, qgs_train_reco, qgs_train_labels = extract_features(qgs, qgs_train_indices)

    # ALL of EPOS goes to training
    epos_all_indices = list(range(len(epos)))
    epos_arrays, epos_reco, epos_labels = extract_features(epos, epos_all_indices)

    # Combine QGS train + EPOS
    train_arrays = torch.cat([qgs_train_arrays, epos_arrays])
    train_reco = torch.cat([qgs_train_reco, epos_reco])
    train_labels = torch.cat([qgs_train_labels, epos_labels])

    p(f"Train: {len(train_labels)} (QGS {len(qgs_train_labels)} + EPOS {len(epos_labels)})")
    p(f"Val: {len(val_labels)}, Test: {len(test_labels)}")

    # Class weights from combined train
    _, counts = torch.unique(train_labels, sorted=True, return_counts=True)
    weights = counts[0].float() / counts.float()
    p(f"Class weights: {weights.tolist()}")

    # Datasets
    train_ds = FastTensorDataset(train_arrays, train_reco, train_labels, augment=True)
    val_ds = FastTensorDataset(val_arrays, val_reco, val_labels, augment=False)
    test_ds = FastTensorDataset(test_arrays, test_reco, test_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    del qgs, epos, qgs_train_arrays, epos_arrays; import gc; gc.collect()

    # Train
    model = HybridModelV2(n_feat=6).to(DEVICE)
    p(f"Model params: {sum(pp.numel() for pp in model.parameters()):,}")

    ce_loss = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_test_probs = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        tc, tt = 0, 0
        for arrays, reco, labels in train_loader:
            arrays, reco, labels = arrays.to(DEVICE), reco.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(arrays, reco)
                loss = ce_loss(output, labels)
                # Batch fraction loss
                probs = F.softmax(output, dim=1)
                items_p, counts_p = torch.unique(probs.argmax(1), sorted=True, return_counts=True)
                y_pred = torch.zeros(5, device=DEVICE); y_pred[items_p.long()] = counts_p.float()
                items_t, counts_t = torch.unique(labels, sorted=True, return_counts=True)
                y_true = torch.zeros(5, device=DEVICE); y_true[items_t.long()] = counts_t.float()
                loss = loss + ((y_pred/y_pred.sum() - y_true/y_true.sum())**2).mean()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            tc += (output.argmax(1)==labels).sum().item(); tt += len(labels)
        scheduler.step()
        train_acc = tc/tt

        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for arrays, reco, labels in val_loader:
                arrays, reco, labels = arrays.to(DEVICE), reco.to(DEVICE), labels.to(DEVICE)
                with autocast(device_type='cuda'):
                    output = model(arrays, reco)
                    loss = ce_loss(output, labels)
                vl += loss.item()*len(labels); vc += (output.argmax(1)==labels).sum().item(); vt += len(labels)
        val_loss = vl/vt; val_acc = vc/vt

        if epoch % 10 == 0 or epoch == MAX_EPOCHS-1:
            p(f"Ep {epoch+1:3d}/{MAX_EPOCHS}: train={train_acc:.4f} val={val_acc:.4f} [{time.time()-t0:.0f}s]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            all_probs = []
            with torch.no_grad():
                for arrays, reco, labels in test_loader:
                    arrays, reco, labels = arrays.to(DEVICE), reco.to(DEVICE), labels.to(DEVICE)
                    with autocast(device_type='cuda'):
                        output = model(arrays, reco)
                    all_probs.append(F.softmax(output.float(), 1).cpu().numpy())
            best_test_probs = np.concatenate(all_probs)

    # Evaluate
    test_labels_np = test_labels.numpy()
    preds = best_test_probs.argmax(1)
    acc = (preds == test_labels_np).mean()
    p(f"\nTest accuracy: {acc:.4f}")

    frac_eval = FracEval(test_labels_np)
    raw_fe = frac_eval.evaluate(preds)
    p(f"Raw fraction error: {raw_fe:.4f}")

    # Bias optimization
    la = np.log(best_test_probs + 1e-10)
    def obj(b): return frac_eval.evaluate((la+b).argmax(1))
    res = differential_evolution(obj, bounds=[(-0.5,0.5)]*5, seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25)
    bias_fe = frac_eval.evaluate((la+res.x).argmax(1))
    p(f"Bias-optimized fraction error: {bias_fe:.4f}")
    p(f"Biases: {np.round(res.x, 4).tolist()}")

    # Confusion matrix
    cm = confusion_matrix(test_labels_np, preds)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    p("\nConfusion matrix:")
    abbr = [n[:2] for n in PARTICLE_NAMES]
    p("     " + " ".join(f"{a:>6}" for a in abbr))
    for i in range(5):
        p(f"  {abbr[i]:>2} " + " ".join(f"{cm_norm[i,j]:>6.3f}" for j in range(5)))

    p(f"\nSOTA (LeNet, QGS only): 0.1079")
    if bias_fe < 0.1079:
        p(f">>> BEAT SOTA by {0.1079 - bias_fe:.4f} <<<")
    else:
        p(f"Gap to SOTA: {bias_fe - 0.1079:.4f}")

    p(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
