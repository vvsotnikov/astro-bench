"""Retrain v8 with SEED=42 (matching the original Opus run) for reproducibility check.

If this run reproduces the original 0.1053 result, the seed=1 delta of 4e-3
reflects real training-seed stochasticity. If this run diverges substantially
from 0.1053, something else (Python env, CUDA non-determinism, DataLoader
worker randomness) is driving the spread, and the seed=1 result is not a
clean seed-stochasticity measurement.

Run with:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python paper/retrain_v8_seed42.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "composition"))
from load_data import load_train, load_test  # noqa: E402
from verify import _compute_fraction_error  # noqa: E402


DEVICE = "cuda"
BATCH_SIZE = 1024
MAX_EPOCHS = 150
LR = 1e-3
SEED = 42  # <-- same seed as original Opus run
OUT_MODEL = HERE / "model_v8_s42_reproduce.pt"
OUT_PREDS = HERE / "predictions_v8_s42_reproduce.npz"


def p(msg):
    print(msg, flush=True)


class CompositionDataset(Dataset):
    def __init__(self, matrices, features, labels, augment=False):
        self.matrices = matrices
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = torch.from_numpy(self.matrices[idx].copy().astype(np.float32))
        feat = self.features[idx].astype(np.float32)
        label = int(self.labels[idx])

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                mat = torch.rot90(mat, k, [0, 1])
            if torch.rand(1) < 0.5:
                mat = torch.flip(mat, dims=[0])
            if torch.rand(1) < 0.5:
                mat = torch.flip(mat, dims=[1])

        E = (feat[0] - 16.0) / 1.0
        Ze = feat[1] / 30.0
        Az = feat[2] / 360.0
        Ne = (feat[3] - 5.31) / 0.5
        Nmu = (feat[4] - 4.3) / 0.42
        Age = feat[5] - 1.0
        Ne_Nmu = feat[3] - feat[4]
        Ne_Nmu_r = feat[3] / (feat[4] + 1e-6)

        reco = torch.tensor(
            [E, Ze, Az, Ne, Nmu, Age, Ne_Nmu, Ne_Nmu_r], dtype=torch.float32
        )
        return mat, reco, label


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class CompositionNet(nn.Module):
    def __init__(self, n_classes=5, n_reco=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.layer1 = nn.Sequential(ResBlock(32), ResBlock(32), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), ResBlock(64), nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128), nn.AdaptiveAvgPool2d(1),
        )
        self.feat_net = nn.Sequential(
            nn.Linear(n_reco, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
        self.cnn_head = nn.Linear(128, n_classes)

    def forward(self, x, x_reco, return_both=False):
        x = x.permute(0, 3, 1, 2).float()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        cnn_feat = x.flatten(1)
        f = self.feat_net(x_reco)
        out = self.classifier(torch.cat([cnn_feat, f], dim=1))
        if return_both:
            return out, self.cnn_head(cnn_feat)
        return out


def batch_fraction_loss(logits, targets, n_classes=5):
    probs = F.softmax(logits, dim=1)
    pred_counts = torch.zeros(n_classes, device=logits.device)
    ip, cp = torch.unique(probs.argmax(1), sorted=True, return_counts=True)
    pred_counts[ip.long()] = cp.float()

    true_counts = torch.zeros(n_classes, device=logits.device)
    it, ct = torch.unique(targets, sorted=True, return_counts=True)
    true_counts[it.long()] = ct.float()

    pred_frac = pred_counts / pred_counts.sum()
    true_frac = true_counts / true_counts.sum()
    return ((pred_frac - true_frac) ** 2).mean()


def mixup_data(x_mat, x_reco, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x_mat.size(0), device=x_mat.device)
    return (
        lam * x_mat + (1 - lam) * x_mat[idx],
        lam * x_reco + (1 - lam) * x_reco[idx],
        y, y[idx], lam,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    p(f"SEED = {SEED} (reproduction attempt of Opus v8 = 0.1053)")
    p("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()

    full_train = CompositionDataset(X_train, f_train, y_train, augment=True)

    n = len(full_train)
    n_val = int(n * 0.15)
    n_trn = n - n_val
    trn_ds, val_indices = torch.utils.data.random_split(
        full_train, [n_trn, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_raw = CompositionDataset(X_train, f_train, y_train, augment=False)
    val_ds = torch.utils.data.Subset(val_raw, val_indices.indices)

    train_loader = DataLoader(
        trn_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    labels_arr = np.array(y_train)
    _, counts = np.unique(labels_arr, return_counts=True)
    weights = torch.tensor(
        counts.sum() / (5 * counts), dtype=torch.float32
    ).to(DEVICE)

    model = CompositionNet().to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"  Params: {n_params:,}")

    ce = nn.CrossEntropyLoss(weight=weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        t_epoch = time.time()
        model.train()
        tc, tt = 0, 0
        for mat, reco, y in train_loader:
            mat, reco, y = mat.to(DEVICE), reco.to(DEVICE), y.to(DEVICE)
            mat_m, reco_m, ya, yb, lam = mixup_data(mat, reco, y)

            optimizer.zero_grad()
            out, out_cnn = model(mat_m, reco_m, return_both=True)

            loss_main = lam * ce(out, ya).mean() + (1 - lam) * ce(out, yb).mean()
            loss_cnn = lam * ce(out_cnn, ya).mean() + (1 - lam) * ce(out_cnn, yb).mean()
            loss = loss_main / 5 + loss_cnn / 5
            loss += batch_fraction_loss(out, ya)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tc += (out.argmax(1) == ya).sum().item()
            tt += len(y)

        scheduler.step()

        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for mat, reco, y in val_loader:
                mat, reco, y = mat.to(DEVICE), reco.to(DEVICE), y.to(DEVICE)
                out, out_cnn = model(mat, reco, return_both=True)
                loss = ce(out, y).mean() / 5 + ce(out_cnn, y).mean() / 5
                vl += loss.item() * len(y)
                vc += (out.argmax(1) == y).sum().item()
                vt += len(y)
        val_loss = vl / vt

        if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
            p(
                f"  Ep {epoch+1}/{MAX_EPOCHS}: acc={tc/tt:.4f} val_acc={vc/vt:.4f} "
                f"vl={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"[{time.time()-t_epoch:.0f}s]"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(OUT_MODEL))
        else:
            patience_counter += 1
            if patience_counter >= 20:
                p(f"  Early stop at ep {epoch+1}")
                break

    p(f"  Best val_loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(str(OUT_MODEL), weights_only=True))
    model.eval()

    all_probs = []
    for t in range(8):
        ds = CompositionDataset(X_test, f_test, y_test, augment=(t > 0))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for mat, reco, y in loader:
                mat, reco = mat.to(DEVICE), reco.to(DEVICE)
                out = model(mat, reco)
                probs.append(F.softmax(out, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs))

    avg_probs = np.mean(all_probs, axis=0)
    predictions = avg_probs.argmax(axis=1)
    np.savez(str(OUT_PREDS), predictions=predictions)

    labels = y_test.astype(int)
    res = _compute_fraction_error(labels, predictions.astype(int))
    fe = res["mean"]

    p(f"\n=== v8 seed={SEED} reproduction result ===")
    p(f"Mean fraction error: {fe:.8f}  (rounded: {fe:.4f})")
    p(f"Original Opus v8 (same seed=42): 0.1053  (attempt 7, composition-opus-2apr/v8.log)")
    p(f"Reproduction delta: {abs(fe - 0.1053):.3e}  (|fe - 0.1053|)")
    p(f"v8 seed=1 (this paper/retrain_v8_seed1.py): 0.10925618")
    p(f"Total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
