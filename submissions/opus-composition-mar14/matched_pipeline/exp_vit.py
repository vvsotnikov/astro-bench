"""GPU experiment: Vision Transformer on 16x16 detector grid.
Fundamentally different inductive bias from CNN — global attention from the start.
4x4 patches → 16 tokens → transformer encoder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import time, math

DATA_DIR = "data"
DEVICE = "cuda"
BATCH_SIZE = 2048
MAX_EPOCHS = 80
LR = 5e-4
SEED = 42
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
NORM = {'Ne': (5.31, 0.5), 'Nmu': (4.3, 0.42), 'Age': (1.0, 1.0), 'Ze': (0.0, 60.0)}
MIXTURE_SIZE = 5000; MIXTURE_SEED = 2026; GRID_STEP = 0.1

def p(msg): print(msg, flush=True)

# --- Data loading (same as beat_sota.py) ---
class QGSData:
    def __init__(self):
        raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
        raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
        reco = raw_feat[:, 1:].astype(np.float32)
        mask = np.ones(len(reco), dtype=bool)
        for feat_name, (lo, hi) in CUTS.items():
            idx = RECO_HEADERS.index(feat_name)
            mask &= (reco[:, idx] > lo) & (reco[:, idx] < hi)
        self.arrays = torch.from_numpy(raw_mat[mask])
        self.reco = torch.from_numpy(reco[mask])
        self.labels = torch.from_numpy(raw_feat[mask, 0].astype(np.int64)) - 1
        p(f"  Loaded {len(self.labels)} events")

class FastDS(Dataset):
    def __init__(self, data, indices, augment=False):
        self.augment = augment
        self.x = data.arrays[indices][:, :, :, [1, 2]].float()
        r = data.reco[indices]
        Ne = r[:, 6]; Nmu = r[:, 7]; Age = r[:, 8]; Ze = r[:, 4]
        self.reco = torch.stack([(Ne-5.31)/0.5, (Nmu-4.3)/0.42, Age-1.0, Ze/60.0,
                                  (Ne-Nmu-0.8)/0.3, (r[:, 0]-15.5)/1.0], dim=1)
        self.y = data.labels[indices]
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        a = self.x[idx]
        if self.augment:
            if torch.rand(1) < 0.5: a = torch.rot90(a, 1, [0, 1])
            if torch.rand(1) < 0.5: a = torch.rot90(a, 2, [0, 1])
            if torch.rand(1) < 0.5: a = torch.flip(a, dims=[0])
        return a, self.reco[idx], self.y[idx]

# --- ViT model ---
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_ch=2, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: (B, H, W, C) → (B, C, H, W) → patches
        x = x.permute(0, 3, 1, 2).float()
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTClassifier(nn.Module):
    def __init__(self, patch_size=4, in_ch=2, embed_dim=128, depth=4, n_heads=4,
                 n_reco=6, n_classes=5, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_ch, embed_dim)
        n_patches = (16 // patch_size) ** 2  # 16 for 16x16 grid
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, n_heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.reco_mlp = nn.Sequential(
            nn.Linear(n_reco, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def forward(self, x, reco):
        x = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x[:, 0])  # CLS token
        r = self.reco_mlp(reco)
        return self.head(torch.cat([x, r], dim=1))

# --- Evaluation ---
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
    data = QGSData()
    n = len(data.labels); nv = int(n*0.3); nt = n-nv
    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    train_idx = all_idx[:nt]; valid_idx = all_idx[nt:]
    vperm = torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    ntest = int(nv*0.3); nval = nv-ntest
    val_idx = valid_idx[vperm[:nval]]; test_idx = valid_idx[vperm[nval:]]

    train_ds = FastDS(data, train_idx, augment=True)
    val_ds = FastDS(data, val_idx)
    test_ds = FastDS(data, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    _, counts = torch.unique(train_ds.y, sorted=True, return_counts=True)
    weights = counts[0].float() / counts.float()

    feval = FracEval(test_ds.y.numpy())

    configs = [
        {"embed_dim": 128, "depth": 4, "n_heads": 4, "label": "ViT-128-4"},
        {"embed_dim": 64, "depth": 6, "n_heads": 4, "label": "ViT-64-6"},
    ]

    for cfg in configs:
        torch.manual_seed(SEED)
        model = ViTClassifier(embed_dim=cfg["embed_dim"], depth=cfg["depth"],
                              n_heads=cfg["n_heads"]).to(DEVICE)
        np_ = sum(pp.numel() for pp in model.parameters())
        p(f"\n{cfg['label']}: {np_:,} params")

        ce = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=0.03)
        # Warmup + cosine
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        warmup_epochs = 5
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (MAX_EPOCHS - warmup_epochs)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        scaler = GradScaler()

        best_vl = float('inf'); best_probs = None
        for epoch in range(MAX_EPOCHS):
            model.train()
            for a, r, y in train_loader:
                a, r, y = a.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                with autocast(device_type='cuda'):
                    out = model(a, r)
                loss = ce(out.float(), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()

            if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
                model.eval()
                vl = 0; vt = 0
                all_p = []
                with torch.no_grad():
                    for a, r, y in val_loader:
                        a, r, y = a.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
                        with autocast(device_type='cuda'):
                            out = model(a, r)
                        vl += ce(out.float(), y).item()*len(y); vt += len(y)
                    for a, r, y in test_loader:
                        a, r, y = a.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
                        with autocast(device_type='cuda'):
                            out = model(a, r)
                        all_p.append(F.softmax(out.float(), 1).cpu().numpy())
                test_probs = np.concatenate(all_p)
                preds = test_probs.argmax(1)
                acc = (preds == test_ds.y.numpy()).mean()
                fe = feval.evaluate(preds)
                p(f"  Ep {epoch+1}: acc={acc:.4f} fe={fe:.4f} val_loss={vl/vt:.4f} [{time.time()-t0:.0f}s]")

                if vl/vt < best_vl:
                    best_vl = vl/vt
                    best_probs = test_probs

        preds = best_probs.argmax(1)
        acc = (preds == test_ds.y.numpy()).mean()
        fe = feval.evaluate(preds)
        p(f"  Final: acc={acc:.4f} fe={fe:.4f}")
        np.save(f"experiments/vit_{cfg['label']}_probs.npy", best_probs)

    p(f"\nTotal: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
