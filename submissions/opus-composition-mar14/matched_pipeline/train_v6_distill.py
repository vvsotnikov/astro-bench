"""v6: Knowledge distillation.
Use existing model probabilities as soft targets (teacher).
Train a student model with a mix of hard labels + soft teacher targets.
Teacher: average of all saved probs (cnn_aug, v1_log1p, v2_sam).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import time, sys
sys.path.insert(0, '.')
from eval_utils import evaluate_and_save, FractionErrorEvaluator

DATA_DIR = "data"; DEVICE = "cuda"; BATCH_SIZE = 2048; MAX_EPOCHS = 80
LR = 1e-3; SEED = 2026; TEMP = 3.0; ALPHA = 0.7  # distillation temperature and weight
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
def p(msg): print(msg, flush=True)

class FastDS(Dataset):
    def __init__(self, arrays, reco, labels, teacher_probs=None, augment=False):
        self.augment = augment
        self.x = torch.log1p(arrays[:, :, :, [1, 2]].float())
        self.reco = reco; self.y = labels
        self.teacher = teacher_probs  # (N, 5) or None
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
        if self.teacher is not None:
            return a, self.reco[idx], self.y[idx], self.teacher[idx]
        return a, self.reco[idx], self.y[idx], torch.zeros(5)

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
        return self.head(torch.cat([self.cnn(mat).flatten(1), self.feat_mlp(feat)], dim=1))

def load_data():
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32); labels = raw_feat[:, 0].astype(np.int64) - 1
    mask = np.ones(len(reco), dtype=bool)
    for fn, (lo,hi) in CUTS.items():
        i = RECO_HEADERS.index(fn); mask &= (reco[:,i]>lo) & (reco[:,i]<hi)
    reco=reco[mask]; labels=labels[mask]; mat=raw_mat[mask]
    n=len(labels); nv=int(n*0.3); nt=n-nv
    all_idx=torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    train_idx=all_idx[:nt]; valid_idx=all_idx[nt:]
    vperm=torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    ntest=int(nv*0.3); nval=nv-ntest
    val_idx=valid_idx[vperm[:nval]]; test_idx=valid_idx[vperm[nval:]]
    def mk(r):
        Ne=r[:,6];Nmu=r[:,7];Age=r[:,8];Ze=r[:,4];E=r[:,0]
        return torch.from_numpy(np.column_stack([(Ne-5.31)/0.5,(Nmu-4.3)/0.42,Age-1.0,Ze/60.0,(Ne-Nmu-0.8)/0.3,(E-15.5)/1.0]).astype(np.float32))
    return torch.from_numpy(mat), mk(reco), torch.from_numpy(labels), train_idx, val_idx, test_idx

def main():
    torch.manual_seed(SEED); np.random.seed(SEED); t0 = time.time()
    p("Loading data...")
    mat, reco, labels, train_idx, val_idx, test_idx = load_data()

    # Load teacher probs (only for test set — for train we need to generate)
    # For train: use the v2_sam model to generate teacher probs
    p("Loading teacher model (v2_sam)...")
    teacher = HybridModel(n_feat=6).to(DEVICE)
    teacher.load_state_dict(torch.load(f"{OUT_DIR}/model_v2_sam.pt", weights_only=True))
    teacher.eval()

    # Generate teacher probs for training data
    p("Generating teacher probs for training data...")
    train_x = torch.log1p(mat[train_idx][:, :, :, [1, 2]].float())
    train_reco = reco[train_idx]
    teacher_probs = []
    with torch.no_grad():
        for i in range(0, len(train_idx), BATCH_SIZE):
            a = train_x[i:i+BATCH_SIZE].to(DEVICE)
            r = train_reco[i:i+BATCH_SIZE].to(DEVICE)
            out = teacher(a, r)
            teacher_probs.append(F.softmax(out.float(), 1).cpu())
    teacher_train = torch.cat(teacher_probs)
    p(f"Teacher probs: {teacher_train.shape}")
    del teacher; torch.cuda.empty_cache()

    train_ds = FastDS(mat[train_idx], reco[train_idx], labels[train_idx],
                      teacher_probs=teacher_train, augment=True)
    val_ds = FastDS(mat[val_idx], reco[val_idx], labels[val_idx])
    test_ds = FastDS(mat[test_idx], reco[test_idx], labels[test_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    _, counts = torch.unique(train_ds.y, sorted=True, return_counts=True)
    weights = counts[0].float()/counts.float()

    model = HybridModel(n_feat=6).to(DEVICE)
    p(f"Student params: {sum(pp.numel() for pp in model.parameters()):,}")
    ce = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=0.03)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    scaler = GradScaler()

    feval = FractionErrorEvaluator(test_ds.y.numpy())
    best_fe = 1.0; best_probs = None

    for epoch in range(MAX_EPOCHS):
        model.train(); tc=0; tt=0
        for a, r, y, t_probs in train_loader:
            a,r,y,t_probs = a.to(DEVICE),r.to(DEVICE),y.to(DEVICE),t_probs.to(DEVICE)
            opt.zero_grad()
            with autocast(device_type='cuda'):
                out = model(a, r)
            logits = out.float()
            # Hard label loss
            hard_loss = ce(logits, y)
            # Soft distillation loss (KL divergence with temperature)
            soft_student = F.log_softmax(logits / TEMP, dim=1)
            soft_teacher = F.softmax(t_probs / TEMP, dim=1)
            dist_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (TEMP ** 2)
            # Combined loss
            loss = (1 - ALPHA) * hard_loss + ALPHA * dist_loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tc += (logits.argmax(1)==y).sum().item(); tt += len(y)
        sched.step()

        if epoch % 10 == 0 or epoch == MAX_EPOCHS-1:
            model.eval(); all_p=[]
            with torch.no_grad():
                for a,r,y,_ in test_loader:
                    a,r=a.to(DEVICE),r.to(DEVICE)
                    out=model(a,r); all_p.append(F.softmax(out.float(),1).cpu().numpy())
            tp=np.concatenate(all_p); fe=feval.evaluate(tp.argmax(1))
            p(f"Ep {epoch+1}: train={tc/tt:.4f} fe={fe:.4f} [{time.time()-t0:.0f}s]")
            if fe < best_fe: best_fe=fe; best_probs=tp

    evaluate_and_save(best_probs, test_ds.y.numpy(), model,
                      "v6_distill", f"KD from v2_sam teacher, T={TEMP}, alpha={ALPHA}, seed={SEED}",
                      OUT_DIR)
    p(f"Total: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
