"""Evaluate v3 TTA model (training already done, just redo TTA + DE)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.amp import autocast
import sys, time
sys.path.insert(0, '.')
from eval_utils import evaluate_and_save

DATA_DIR = "data"
DEVICE = "cuda"
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']

def p(msg): print(msg, flush=True)

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

def main():
    t0 = time.time()
    p("Loading data and model...")
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
    valid_idx=all_idx[nt:]
    vperm=torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    ntest=int(nv*0.3)
    test_idx=valid_idx[vperm[nv-ntest:]]

    test_mat = torch.log1p(torch.from_numpy(mat[test_idx][:, :, :, [1, 2]]).float())
    r = reco[test_idx]
    Ne=r[:,6];Nmu=r[:,7];Age=r[:,8];Ze=r[:,4];E=r[:,0]
    test_reco = torch.from_numpy(np.column_stack([
        (Ne-5.31)/0.5,(Nmu-4.3)/0.42,Age-1.0,Ze/60.0,
        (Ne-Nmu-0.8)/0.3,(E-15.5)/1.0]).astype(np.float32))
    test_labels = labels[test_idx]

    model = HybridModel(n_feat=6).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v3_tta_best.pt", weights_only=True))
    model.eval()

    BS = 2048
    # Without TTA
    no_tta = []
    with torch.no_grad():
        for i in range(0, len(test_labels), BS):
            a = test_mat[i:i+BS].to(DEVICE); r_ = test_reco[i:i+BS].to(DEVICE)
            with autocast(device_type='cuda'):
                out = model(a, r_)
            no_tta.append(F.softmax(out.float(), 1).cpu().numpy())
    no_tta = np.concatenate(no_tta)
    p(f"Without TTA: fe={np.mean(np.abs(no_tta.argmax(1) - test_labels)):.4f}")  # rough check

    # TTA: 4 rotations × 2 flips = 8
    all_aug = []
    for rot in range(4):
        for flip in [False, True]:
            aug = []
            with torch.no_grad():
                for i in range(0, len(test_labels), BS):
                    a = test_mat[i:i+BS].to(DEVICE); r_ = test_reco[i:i+BS].to(DEVICE)
                    if rot > 0: a = torch.rot90(a, rot, [1, 2])
                    if flip: a = torch.flip(a, dims=[1])
                    with autocast(device_type='cuda'):
                        out = model(a, r_)
                    aug.append(F.softmax(out.float(), 1).cpu().numpy())
            all_aug.append(np.concatenate(aug))
    tta_probs = np.mean(all_aug, axis=0)

    # Use TTA probs
    evaluate_and_save(tta_probs, test_labels, model,
                      "v3_tta", "CNN+Attn+MLP log1p + aug + TTA(8x)", OUT_DIR)
    p(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
