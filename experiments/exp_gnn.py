"""GNN: Treat active detector stations as graph nodes.
Instead of dense 16x16 grid (85% zeros), build a sparse graph from non-zero cells.
Each node has features: [e_density, mu_density, log1p(e), log1p(mu), x, y].
Edges: k-nearest neighbors in spatial coordinates.
Message passing → global pooling → combine with reco features → classify.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import time, math

DATA_DIR = "data"
DEVICE = "cuda"
BATCH_SIZE = 1024
MAX_EPOCHS = 80
LR = 5e-4
SEED = 42
K = 6  # neighbors per node
MAX_NODES = 48
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
MIXTURE_SIZE = 5000; MIXTURE_SEED = 2026; GRID_STEP = 0.1

def p(msg): print(msg, flush=True)

# --- Data ---
class GraphDataset(Dataset):
    def __init__(self, arrays, reco_feats, labels, augment=False):
        self.labels = labels
        self.reco = reco_feats
        self.augment = augment
        # Pre-convert matrices to graph representation
        n = len(labels)
        self.node_feat = np.zeros((n, MAX_NODES, 6), dtype=np.float32)
        self.edge_idx = np.zeros((n, MAX_NODES, K), dtype=np.int64)
        self.mask = np.zeros((n, MAX_NODES), dtype=np.float32)

        p(f"  Building graphs for {n} events...")
        for i in range(n):
            mat = arrays[i]  # (16, 16, 3) - ch0=timing, ch1=e/gamma, ch2=muon
            ch_e = mat[:, :, 1].astype(np.float32)
            ch_m = mat[:, :, 2].astype(np.float32)
            nonzero = (ch_e > 0) | (ch_m > 0)
            ys, xs = np.where(nonzero)
            nn_ = len(ys)
            if nn_ == 0:
                continue
            if nn_ > MAX_NODES:
                energy = ch_e[nonzero] + ch_m[nonzero]
                top = np.argsort(energy)[-MAX_NODES:]
                ys, xs = ys[top], xs[top]
                nn_ = MAX_NODES

            for j in range(nn_):
                y_, x_ = ys[j], xs[j]
                self.node_feat[i, j] = [
                    ch_e[y_, x_], ch_m[y_, x_],
                    np.log1p(ch_e[y_, x_]), np.log1p(ch_m[y_, x_]),
                    x_ / 15.0, y_ / 15.0
                ]
                self.mask[i, j] = 1.0

            if nn_ > 1:
                pos = np.column_stack([xs, ys]).astype(float)
                for j in range(nn_):
                    dists = np.sqrt(((pos - pos[j])**2).sum(axis=1))
                    dists[j] = 1e10
                    k_actual = min(K, nn_ - 1)
                    neighbors = np.argsort(dists)[:k_actual]
                    self.edge_idx[i, j, :k_actual] = neighbors
                    self.edge_idx[i, j, k_actual:] = j

            if i % 50000 == 0 and i > 0:
                p(f"    {i}/{n}")

        self.node_feat = torch.from_numpy(self.node_feat)
        self.edge_idx = torch.from_numpy(self.edge_idx)
        self.mask = torch.from_numpy(self.mask)
        p(f"  Graphs built.")

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.node_feat[idx], self.edge_idx[idx], self.mask[idx], self.reco[idx], self.labels[idx]

# --- Model ---
class MPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 2, out_dim), nn.ReLU())
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, x, pos, edge_idx, mask):
        B, N, D = x.shape; K_ = edge_idx.shape[2]
        ei_flat = edge_idx.reshape(B, -1)
        nbrs = torch.gather(x, 1, ei_flat.unsqueeze(-1).expand(-1, -1, D)).reshape(B, N, K_, D)
        nbr_pos = torch.gather(pos, 1, ei_flat.unsqueeze(-1).expand(-1, -1, 2)).reshape(B, N, K_, 2)
        rel_pos = nbr_pos - pos.unsqueeze(2).expand(-1, -1, K_, -1)
        x_exp = x.unsqueeze(2).expand(-1, -1, K_, -1)
        edge_feat = torch.cat([x_exp, nbrs, rel_pos], dim=-1)
        messages = self.edge_mlp(edge_feat).mean(dim=2)
        updated = self.node_mlp(torch.cat([x, messages], dim=-1))
        return updated * mask.unsqueeze(-1)

class GNNClassifier(nn.Module):
    def __init__(self, node_dim=6, hidden=64, n_reco=6, n_classes=5):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(node_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mp1 = MPLayer(hidden, hidden)
        self.mp2 = MPLayer(hidden, hidden)
        self.mp3 = MPLayer(hidden, hidden)
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.reco_mlp = nn.Sequential(
            nn.Linear(n_reco, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + 64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def forward(self, nf, ei, mask, reco):
        pos = nf[:, :, 4:6]  # x, y positions
        x = self.embed(nf)
        x = self.mp1(x, pos, ei, mask)
        x = self.mp2(x, pos, ei, mask)
        x = self.mp3(x, pos, ei, mask)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        graph = (x * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        graph = self.readout(graph)
        r = self.reco_mlp(reco)
        return self.head(torch.cat([graph, r], dim=-1))

# --- Eval ---
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
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32)
    labels_all = raw_feat[:, 0].astype(np.int64) - 1
    mask = np.ones(len(reco), dtype=bool)
    for feat_name, (lo, hi) in CUTS.items():
        idx = RECO_HEADERS.index(feat_name)
        mask &= (reco[:, idx] > lo) & (reco[:, idx] < hi)
    reco = reco[mask]; labels_all = labels_all[mask]; mat = raw_mat[mask]
    p(f"  {len(labels_all)} events after cuts")

    # Split
    n = len(labels_all); nv = int(n*0.3); nt = n-nv
    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    train_idx = all_idx[:nt]; valid_idx = all_idx[nt:]
    vperm = torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    ntest = int(nv*0.3); nval = nv-ntest
    val_idx = valid_idx[vperm[:nval]]; test_idx = valid_idx[vperm[nval:]]

    # Reco features
    def make_reco(r):
        Ne=r[:,6]; Nmu=r[:,7]; Age=r[:,8]; Ze=r[:,4]; E=r[:,0]
        return torch.from_numpy(np.column_stack([
            (Ne-5.31)/0.5, (Nmu-4.3)/0.42, Age-1.0, Ze/60.0,
            (Ne-Nmu-0.8)/0.3, (E-15.5)/1.0
        ]).astype(np.float32))

    # Build graph datasets (subsample train for speed)
    max_train = min(len(train_idx), 200000)
    train_sub = train_idx[:max_train]
    p(f"Train: {max_train}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_ds = GraphDataset(mat[train_sub], make_reco(reco[train_sub]),
                            torch.from_numpy(labels_all[train_sub]))
    val_ds = GraphDataset(mat[val_idx], make_reco(reco[val_idx]),
                          torch.from_numpy(labels_all[val_idx]))
    test_ds = GraphDataset(mat[test_idx], make_reco(reco[test_idx]),
                           torch.from_numpy(labels_all[test_idx]))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    _, counts = torch.unique(torch.from_numpy(labels_all[train_sub]), sorted=True, return_counts=True)
    weights = counts[0].float() / counts.float()

    model = GNNClassifier(hidden=64, n_reco=6).to(DEVICE)
    p(f"Params: {sum(pp.numel() for pp in model.parameters()):,}")

    ce = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=0.03)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warmup = 5
    def lr_fn(ep):
        if ep < warmup: return ep / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (MAX_EPOCHS - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    scaler = GradScaler()

    feval = FracEval(test_ds.labels.numpy())
    best_vl = float('inf'); best_probs = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for nf, ei, mk, rc, y in train_loader:
            nf, ei, mk, rc, y = nf.to(DEVICE), ei.to(DEVICE), mk.to(DEVICE), rc.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with autocast(device_type='cuda'):
                out = model(nf, ei, mk, rc)
            loss = ce(out.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()

        if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
            model.eval()
            vl = 0; vt = 0; all_p = []
            with torch.no_grad():
                for nf, ei, mk, rc, y in val_loader:
                    nf, ei, mk, rc, y = nf.to(DEVICE), ei.to(DEVICE), mk.to(DEVICE), rc.to(DEVICE), y.to(DEVICE)
                    out = model(nf, ei, mk, rc)
                    vl += ce(out.float(), y).item()*len(y); vt += len(y)
                for nf, ei, mk, rc, y in test_loader:
                    nf, ei, mk, rc, y = nf.to(DEVICE), ei.to(DEVICE), mk.to(DEVICE), rc.to(DEVICE), y.to(DEVICE)
                    out = model(nf, ei, mk, rc)
                    all_p.append(F.softmax(out.float(), 1).cpu().numpy())
            tp = np.concatenate(all_p)
            preds = tp.argmax(1)
            acc = (preds == test_ds.labels.numpy()).mean()
            fe = feval.evaluate(preds)
            p(f"Ep {epoch+1}: acc={acc:.4f} fe={fe:.4f} [{time.time()-t0:.0f}s]")
            if vl/vt < best_vl:
                best_vl = vl/vt; best_probs = tp

    preds = best_probs.argmax(1)
    fe = feval.evaluate(preds)
    acc = (preds == test_ds.labels.numpy()).mean()
    p(f"\nFinal: acc={acc:.4f} fe={fe:.4f}")
    np.save("experiments/gnn_probs.npy", best_probs)
    p(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
