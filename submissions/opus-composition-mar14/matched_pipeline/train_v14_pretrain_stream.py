"""v14: Streaming pre-training on real KASCADE data.
Instead of downloading all 73M events to RAM, stream runs directly:
- Download one run at a time from S3
- Train MAE on that run's data
- Discard and move to next run
Constant memory usage regardless of total data size.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import boto3, io, time, sys, gc
from botocore import UNSIGNED
from botocore.config import Config
sys.path.insert(0, '.')
from eval_utils import evaluate_and_save, FractionErrorEvaluator

DATA_DIR = "data"; DEVICE = "cuda"; SEED = 42
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
BUCKET = 'kascade-with-arr-times-npz'
PRETRAIN_PASSES = 2  # number of passes over all runs
PRETRAIN_BS = 4096
FINETUNE_EPOCHS = 80
FINETUNE_BS = 2048
LR_PRETRAIN = 1e-3
LR_FINETUNE = 1e-3

def p(msg): print(msg, flush=True)

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch//r,8)), nn.ReLU(), nn.Linear(max(ch//r,8), ch), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(2,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256))
    def forward(self, x): return self.block3(self.block2(self.block1(x)))

class MAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 2, 1))
    def forward(self, x): return self.decoder(x)

class HybridClassifier(nn.Module):
    def __init__(self, encoder, n_feat=6, n_classes=5):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(
            nn.Linear(256+128,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,n_classes))
    def forward(self, mat, feat):
        mat = mat.permute(0,3,1,2)
        return self.head(torch.cat([self.pool(self.encoder(mat)).flatten(1), self.feat_mlp(feat)], dim=1))

class FastDS(torch.utils.data.Dataset):
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
            if torch.rand(1) < 0.3: a = a + torch.randn_like(a) * 0.1 * a.std()
        return a, self.reco[idx], self.y[idx]

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    @torch.no_grad()
    def first_step(self):
        gn = torch.stack([pp.grad.norm(2) for g in self.param_groups for pp in g["params"] if pp.grad is not None]).norm(2)
        for group in self.param_groups:
            s = group["rho"] / (gn + 1e-12)
            for pp in group["params"]:
                if pp.grad is None: continue
                e = pp.grad * s; pp.add_(e); self.state[pp]["e_w"] = e
    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for pp in group["params"]:
                if pp.grad is None: continue
                pp.sub_(self.state[pp]["e_w"])
        self.base_optimizer.step()
    def zero_grad(self): self.base_optimizer.zero_grad()

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()

    # ===== Phase 1: Streaming MAE pre-training =====
    p("Phase 1: Streaming MAE pre-training from S3...")
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    all_files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET):
        all_files.extend([obj['Key'] for obj in page.get('Contents', [])])
    runs = sorted(set(f.split('_')[0] for f in all_files))
    runs = [r for r in runs if r[-1] in {'0', '1'}]
    p(f"  {len(runs)} runs available")

    encoder = CNNEncoder().to(DEVICE)
    decoder = MAEDecoder().to(DEVICE)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=LR_PRETRAIN, weight_decay=1e-4)
    scaler = GradScaler()
    total_events = 0; total_loss = 0; n_batches = 0

    for pass_num in range(PRETRAIN_PASSES):
        rng = np.random.default_rng(SEED + pass_num)
        run_order = rng.permutation(runs).tolist()

        for ri, run in enumerate(run_order):
            try:
                buf = io.BytesIO()
                s3.download_fileobj(BUCKET, f'{run}_matrices.npz', buf)
                buf.seek(0)
                mat = np.load(buf, allow_pickle=True)['matrices']
                # Use channels 1,2, apply log1p, convert to (N, 2, 16, 16)
                x = torch.log1p(torch.from_numpy(mat[:, :, :, [1, 2]].astype(np.float32))).permute(0, 3, 1, 2)
                del mat

                # Train on this run's data
                encoder.train(); decoder.train()
                indices = torch.randperm(len(x))
                for bi in range(0, len(x), PRETRAIN_BS):
                    batch = x[indices[bi:bi+PRETRAIN_BS]].to(DEVICE)
                    B = batch.shape[0]
                    if B < 16: continue

                    # Random patch masking
                    mask = torch.ones(B, 1, 16, 16, device=DEVICE)
                    for _ in range(8):
                        cx = torch.randint(0, 4, (B,)) * 4
                        cy = torch.randint(0, 4, (B,)) * 4
                        for b in range(B):
                            mask[b, :, cy[b]:cy[b]+4, cx[b]:cx[b]+4] = 0

                    opt.zero_grad()
                    with autocast(device_type='cuda'):
                        features = encoder(batch * mask)
                        reconstruction = decoder(features)
                        loss = F.mse_loss(reconstruction * (1 - mask), batch * (1 - mask))
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    total_loss += loss.item()
                    n_batches += 1

                total_events += len(x)
                del x; torch.cuda.empty_cache()

                if (ri + 1) % 100 == 0:
                    avg_loss = total_loss / max(n_batches, 1)
                    p(f"  Pass {pass_num+1}/{PRETRAIN_PASSES} Run {ri+1}/{len(runs)}: "
                      f"{total_events:,} events, loss={avg_loss:.4f} [{time.time()-t0:.0f}s]")
                    total_loss = 0; n_batches = 0
            except Exception as e:
                continue

        p(f"  Pass {pass_num+1} complete: {total_events:,} total events [{time.time()-t0:.0f}s]")

    torch.save(encoder.state_dict(), f"{OUT_DIR}/encoder_v14_pretrained.pt")
    del decoder; gc.collect(); torch.cuda.empty_cache()
    p(f"Pre-training done: {total_events:,} events [{time.time()-t0:.0f}s]")

    # ===== Phase 2: Fine-tune =====
    p("\nPhase 2: Fine-tuning...")
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32); labels = raw_feat[:, 0].astype(np.int64) - 1
    mask_c = np.ones(len(reco), dtype=bool)
    for fn, (lo,hi) in CUTS.items():
        i = RECO_HEADERS.index(fn); mask_c &= (reco[:,i]>lo) & (reco[:,i]<hi)
    reco=reco[mask_c]; labels=labels[mask_c]; mat=raw_mat[mask_c]
    del raw_mat, raw_feat; gc.collect()

    n=len(labels); nv=int(n*0.3); nt=n-nv
    all_idx=torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    train_idx=all_idx[:nt]; valid_idx=all_idx[nt:]
    vperm=torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    ntest=int(nv*0.3); nval=nv-ntest
    val_idx=valid_idx[vperm[:nval]]; test_idx=valid_idx[vperm[nval:]]

    def mk_reco(r):
        Ne=r[:,6];Nmu=r[:,7];Age=r[:,8];Ze=r[:,4];E=r[:,0]
        return torch.from_numpy(np.column_stack([(Ne-5.31)/0.5,(Nmu-4.3)/0.42,Age-1.0,Ze/60.0,(Ne-Nmu-0.8)/0.3,(E-15.5)/1.0]).astype(np.float32))

    train_ds = FastDS(torch.from_numpy(mat[train_idx]), mk_reco(reco[train_idx]),
                      torch.from_numpy(labels[train_idx]), augment=True)
    test_ds = FastDS(torch.from_numpy(mat[test_idx]), mk_reco(reco[test_idx]),
                     torch.from_numpy(labels[test_idx]))
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BS, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=FINETUNE_BS, shuffle=False, num_workers=0, pin_memory=True)

    _, counts = torch.unique(train_ds.y, sorted=True, return_counts=True)
    weights = counts[0].float()/counts.float()

    model = HybridClassifier(encoder, n_feat=6).to(DEVICE)
    p(f"  Params: {sum(pp.numel() for pp in model.parameters()):,}")
    ce = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=0.05)
    sam = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=LR_FINETUNE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(sam.base_optimizer, T_max=FINETUNE_EPOCHS)

    best_fe = 1.0; best_probs = None
    feval = FractionErrorEvaluator(test_ds.y.numpy())

    for epoch in range(FINETUNE_EPOCHS):
        model.train(); tc=0; tt=0
        for a, r, y in train_loader:
            a,r,y = a.to(DEVICE),r.to(DEVICE),y.to(DEVICE)
            out = model(a, r); loss = ce(out.float(), y)
            pr = F.softmax(out.float(), dim=1)
            ip,cp = torch.unique(pr.argmax(1), sorted=True, return_counts=True)
            yp = torch.zeros(5, device=DEVICE); yp[ip.long()] = cp.float()
            it,ct = torch.unique(y, sorted=True, return_counts=True)
            yt = torch.zeros(5, device=DEVICE); yt[it.long()] = ct.float()
            loss = loss + ((yp/yp.sum() - yt/yt.sum())**2).mean()
            loss.backward(); sam.first_step(); sam.zero_grad()
            out2 = model(a, r); loss2 = ce(out2.float(), y); loss2.backward()
            sam.second_step(); sam.zero_grad()
            tc += (out.argmax(1)==y).sum().item(); tt += len(y)
        sched.step()

        if epoch % 10 == 0 or epoch == FINETUNE_EPOCHS-1:
            model.eval(); all_p=[]
            with torch.no_grad():
                for a,r,y in test_loader:
                    a,r=a.to(DEVICE),r.to(DEVICE)
                    out=model(a,r); all_p.append(F.softmax(out.float(),1).cpu().numpy())
            tp=np.concatenate(all_p); fe=feval.evaluate(tp.argmax(1))
            p(f"  Finetune Ep {epoch+1}: train={tc/tt:.4f} fe={fe:.4f} [{time.time()-t0:.0f}s]")
            if fe < best_fe: best_fe=fe; best_probs=tp

    evaluate_and_save(best_probs, test_ds.y.numpy(), model,
                      "v14_pretrain_stream",
                      f"Streaming MAE pretrain on {total_events:,} real events ({PRETRAIN_PASSES} passes) + SAM finetune",
                      OUT_DIR)
    p(f"Total: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
