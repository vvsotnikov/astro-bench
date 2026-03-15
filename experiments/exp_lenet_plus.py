"""GPU experiment: Enhanced LeNet with more features and longer training.
The published LeNet uses 4 reco features. Let's give it 6+ and see if that helps.
Also try label smoothing and cosine annealing.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import time

DATA_DIR = "data"
DEVICE = "cuda"
BATCH_SIZE = 1024
MAX_EPOCHS = 100
SEED = 42
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
MIXTURE_SIZE = 5000; MIXTURE_SEED = 2026; GRID_STEP = 0.1

def p(msg): print(msg, flush=True)

class QGSDataset(Dataset):
    def __init__(self, mat_file, feat_file, true_file, cut_data=None):
        raw_feat = np.load(feat_file)['features']
        raw_mat = np.load(mat_file)['matrices']
        reco = raw_feat[:, 1:].astype(np.float32)
        labels = raw_feat[:, 0].astype(np.float32)
        self.arrays = torch.from_numpy(raw_mat)
        self.reconstructed = torch.from_numpy(reco)
        self.labels = torch.from_numpy(labels)
        if cut_data:
            mask = torch.ones(len(self.reconstructed), dtype=torch.bool)
            for feat_name, (lo, hi) in cut_data.items():
                idx = RECO_HEADERS.index(feat_name)
                mask &= (self.reconstructed[:, idx] > lo) & (self.reconstructed[:, idx] < hi)
            self.arrays = self.arrays[mask]
            self.reconstructed = self.reconstructed[mask]
            self.labels = self.labels[mask]
        p(f"  Loaded {len(self)} events")
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return idx

class FastDS(Dataset):
    def __init__(self, ds, indices, augment=False):
        self.augment = augment
        arr = ds.arrays[indices][:, :, :, [1, 2]].float()
        r = ds.reconstructed[indices].float()
        Ne = r[:, 6]; Nmu = r[:, 7]; Age = r[:, 8]; Ze = r[:, 4]; E = r[:, 0]
        self.x = arr
        self.reco = torch.stack([
            (Ne - 5.31)/0.5, (Nmu - 4.3)/0.42, Age - 1.0, Ze/60.0,
            (Ne - Nmu - 0.8)/0.3,  # Ne-Nmu ratio
            (E - 15.5)/1.0,        # energy
        ], dim=1)
        self.y = ds.labels[indices].long() - 1
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        a = self.x[idx]
        if self.augment:
            if torch.rand(1) < 0.5: a = torch.rot90(a, 1, [0, 1])
            if torch.rand(1) < 0.5: a = torch.rot90(a, 2, [0, 1])
            if torch.rand(1) < 0.5: a = torch.flip(a, dims=[0])
        return a, self.reco[idx], self.y[idx]

class EnhancedLeNet(nn.Module):
    """LeNet with more channels + BatchNorm + more reco features."""
    def __init__(self, n_reco=6, n_classes=5, ch=32):
        super().__init__()
        self.conv0 = nn.Conv2d(2, ch, 2); self.bn0 = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3); self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 4); self.bn2 = nn.BatchNorm2d(ch)
        self.conv3 = nn.Conv2d(ch, ch, 2); self.bn3 = nn.BatchNorm2d(ch)
        self.fc1 = nn.Linear(ch*3*3 + n_reco, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, x_reco):
        x = x.permute(0, 3, 1, 2).float()
        x = F.max_pool2d(F.relu(self.bn0(self.conv0(x))), 2, stride=1)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2, stride=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.cat((x, x_reco), dim=1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

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
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()

    p("Loading data...")
    ds = QGSDataset(f"{DATA_DIR}/qgs_spectra_matrices.npz",
                    f"{DATA_DIR}/qgs_spectra_features.npz",
                    f"{DATA_DIR}/qgs_spectra_true_features.npz", cut_data=CUTS)

    n = len(ds); nv = int(n*0.3); nt = n - nv
    train_sub, valid_sub = random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(42))
    ntest = int(nv*0.3); nval = nv - ntest
    val_sub, test_sub = random_split(valid_sub, [nval, ntest], generator=torch.Generator().manual_seed(42))

    def unwrap(sub):
        idx = list(range(len(sub))); d = sub
        while hasattr(d, 'dataset'):
            if hasattr(d, 'indices'): idx = [d.indices[i] for i in idx]
            d = d.dataset
        return idx

    train_fast = FastDS(ds, unwrap(train_sub), augment=True)
    val_fast = FastDS(ds, unwrap(val_sub), augment=False)
    test_fast = FastDS(ds, unwrap(test_sub), augment=False)

    train_loader = DataLoader(train_fast, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_fast, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_fast, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    _, counts = torch.unique(train_fast.y, sorted=True, return_counts=True)
    weights = counts[0].float() / counts.float()

    configs = [
        {"ch": 16, "lr": 3e-4, "label": "LeNet-16 (original size)"},
        {"ch": 32, "lr": 1e-3, "label": "LeNet-32 (wider)"},
        {"ch": 32, "lr": 3e-4, "label": "LeNet-32 lr=3e-4"},
        {"ch": 48, "lr": 5e-4, "label": "LeNet-48"},
    ]

    feval = FracEval(test_fast.y.numpy())
    best_overall_fe = 1.0

    for cfg in configs:
        torch.manual_seed(SEED)
        model = EnhancedLeNet(n_reco=6, ch=cfg["ch"]).to(DEVICE)
        np_ = sum(pp.numel() for pp in model.parameters())
        p(f"\n{cfg['label']}: {np_:,} params")

        ce = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=0.03)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        scaler = GradScaler()

        best_vl = float('inf'); best_probs = None
        for epoch in range(MAX_EPOCHS):
            model.train()
            for a, r, y in train_loader:
                a, r, y = a.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                with autocast(device_type='cuda'):
                    out = model(a, r)
                    loss = ce(out, y)
                    # batch frac loss
                    pr = F.softmax(out, dim=1)
                    ip, cp = torch.unique(pr.argmax(1), sorted=True, return_counts=True)
                    yp = torch.zeros(5, device=DEVICE); yp[ip.long()] = cp.float()
                    it, ct = torch.unique(y, sorted=True, return_counts=True)
                    yt = torch.zeros(5, device=DEVICE); yt[it.long()] = ct.float()
                    loss = loss + ((yp/yp.sum() - yt/yt.sum())**2).mean()
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()

            model.eval()
            vl = 0; vt = 0
            with torch.no_grad():
                for a, r, y in val_loader:
                    a, r, y = a.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
                    with autocast(device_type='cuda'):
                        out = model(a, r)
                        loss = ce(out, y)
                    vl += loss.item()*len(y); vt += len(y)
            if vl/vt < best_vl:
                best_vl = vl/vt
                all_p = []
                with torch.no_grad():
                    for a, r, y in test_loader:
                        a, r, y = a.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
                        with autocast(device_type='cuda'):
                            out = model(a, r)
                        all_p.append(F.softmax(out.float(), 1).cpu().numpy())
                best_probs = np.concatenate(all_p)

            if epoch % 20 == 0:
                preds = best_probs.argmax(1)
                acc = (preds == test_fast.y.numpy()).mean()
                fe = feval.evaluate(preds)
                p(f"  Ep {epoch+1}: acc={acc:.4f} fe={fe:.4f} [{time.time()-t0:.0f}s]")

        preds = best_probs.argmax(1)
        acc = (preds == test_fast.y.numpy()).mean()
        fe = feval.evaluate(preds)
        p(f"  Final: acc={acc:.4f} fe={fe:.4f}")
        np.save(f"experiments/lenet_plus_{cfg['ch']}_probs.npy", best_probs)

        if fe < best_overall_fe:
            best_overall_fe = fe

    p(f"\nBest Enhanced LeNet: {best_overall_fe:.4f}")
    p(f"SOTA: 0.107 | Time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
