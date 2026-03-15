"""v2: SAM optimizer (Sharpness-Aware Minimization).
Same architecture as v1 (log1p + augmentation) but with SAM instead of AdamW.
SAM finds flatter minima → better generalization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import time
import sys
sys.path.insert(0, '.')
from eval_utils import evaluate_and_save, FractionErrorEvaluator

DATA_DIR = "data"
DEVICE = "cuda"
BATCH_SIZE = 2048
MAX_EPOCHS = 80
LR = 1e-3
SEED = 42
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']

def p(msg): print(msg, flush=True)

# --- SAM optimizer ---
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization. Wraps a base optimizer."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()

    def _grad_norm(self):
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(2))
        return torch.stack(norms).norm(2)

    def zero_grad(self): self.base_optimizer.zero_grad()
    def step(self): self.base_optimizer.step()

# --- Same model and data as v1 ---
class FastDS(Dataset):
    def __init__(self, arrays, reco, labels, augment=False):
        self.augment = augment
        self.x = torch.log1p(arrays[:, :, :, [1, 2]].float())
        self.reco = reco; self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        a = self.x[idx]
        if self.augment:
            if torch.rand(1) < 0.5: a = torch.rot90(a, 1, [0, 1])
            if torch.rand(1) < 0.5: a = torch.rot90(a, 2, [0, 1])
            if torch.rand(1) < 0.5: a = torch.flip(a, dims=[0])
            if torch.rand(1) < 0.3:
                cx, cy = torch.randint(0, 13, (1,)).item(), torch.randint(0, 13, (1,)).item()
                a = a.clone(); a[cy:cy+4, cx:cx+4, :] = 0
            if torch.rand(1) < 0.3:
                a = a + torch.randn_like(a) * 0.1 * a.std()
        return a, self.reco[idx], self.y[idx]

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

def load_data():
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32)
    labels = raw_feat[:, 0].astype(np.int64) - 1
    mask = np.ones(len(reco), dtype=bool)
    for fn, (lo, hi) in CUTS.items():
        i = RECO_HEADERS.index(fn); mask &= (reco[:, i] > lo) & (reco[:, i] < hi)
    reco=reco[mask]; labels=labels[mask]; mat=raw_mat[mask]
    n=len(labels); nv=int(n*0.3); nt=n-nv
    all_idx=torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    train_idx=all_idx[:nt]; valid_idx=all_idx[nt:]
    vperm=torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    ntest=int(nv*0.3); nval=nv-ntest
    val_idx=valid_idx[vperm[:nval]]; test_idx=valid_idx[vperm[nval:]]
    def mk_reco(r):
        Ne=r[:,6];Nmu=r[:,7];Age=r[:,8];Ze=r[:,4];E=r[:,0]
        return torch.from_numpy(np.column_stack([
            (Ne-5.31)/0.5,(Nmu-4.3)/0.42,Age-1.0,Ze/60.0,
            (Ne-Nmu-0.8)/0.3,(E-15.5)/1.0]).astype(np.float32))
    return (torch.from_numpy(mat), mk_reco(reco), torch.from_numpy(labels),
            train_idx, val_idx, test_idx)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    p("Loading data...")
    mat, reco, labels, train_idx, val_idx, test_idx = load_data()

    train_ds = FastDS(mat[train_idx], reco[train_idx], labels[train_idx], augment=True)
    val_ds = FastDS(mat[val_idx], reco[val_idx], labels[val_idx])
    test_ds = FastDS(mat[test_idx], reco[test_idx], labels[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    _, counts = torch.unique(train_ds.y, sorted=True, return_counts=True)
    weights = counts[0].float()/counts.float()

    model = HybridModel(n_feat=6).to(DEVICE)
    p(f"Params: {sum(pp.numel() for pp in model.parameters()):,}")
    ce = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=0.05)

    # SAM wrapping AdamW
    sam = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(sam.base_optimizer, T_max=MAX_EPOCHS)

    best_vl = float('inf'); best_probs = None
    feval = FractionErrorEvaluator(test_ds.y.numpy())

    for epoch in range(MAX_EPOCHS):
        model.train()
        tc=0; tt=0
        for a, r, y in train_loader:
            a,r,y = a.to(DEVICE),r.to(DEVICE),y.to(DEVICE)

            # SAM first step: compute gradient at current point
            out = model(a, r)
            loss = ce(out.float(), y)
            loss.backward()
            sam.first_step()
            sam.zero_grad()

            # SAM second step: compute gradient at perturbed point
            out2 = model(a, r)
            loss2 = ce(out2.float(), y)
            # Add batch fraction loss
            pr = F.softmax(out2.float(), dim=1)
            ip,cp = torch.unique(pr.argmax(1), sorted=True, return_counts=True)
            yp = torch.zeros(5, device=DEVICE); yp[ip.long()] = cp.float()
            it,ct = torch.unique(y, sorted=True, return_counts=True)
            yt = torch.zeros(5, device=DEVICE); yt[it.long()] = ct.float()
            loss2 = loss2 + ((yp/yp.sum() - yt/yt.sum())**2).mean()
            loss2.backward()
            sam.second_step()
            sam.zero_grad()

            tc += (out.argmax(1)==y).sum().item(); tt += len(y)
        sched.step()

        if epoch % 10 == 0 or epoch == MAX_EPOCHS-1:
            model.eval()
            vl=0; vt_=0; all_p=[]
            with torch.no_grad():
                for a,r,y in val_loader:
                    a,r,y=a.to(DEVICE),r.to(DEVICE),y.to(DEVICE)
                    out=model(a,r); vl+=ce(out.float(),y).item()*len(y); vt_+=len(y)
                for a,r,y in test_loader:
                    a,r,y=a.to(DEVICE),r.to(DEVICE),y.to(DEVICE)
                    out=model(a,r)
                    all_p.append(F.softmax(out.float(),1).cpu().numpy())
            tp=np.concatenate(all_p)
            fe=feval.evaluate(tp.argmax(1))
            p(f"Ep {epoch+1}: train={tc/tt:.4f} fe={fe:.4f} [{time.time()-t0:.0f}s]")
            if vl/vt_ < best_vl: best_vl=vl/vt_; best_probs=tp

    evaluate_and_save(best_probs, test_ds.y.numpy(), model,
                      "v2_sam", "CNN+Attn+MLP log1p + SAM optimizer + augmentation",
                      OUT_DIR)
    p(f"Total: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
