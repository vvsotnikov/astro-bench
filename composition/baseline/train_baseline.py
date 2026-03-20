"""Baseline: LeNet CNN for mass composition (Kuznetsov et al., JINST 2024).

Reproduces the published result using the standardized data pipeline.
Architecture: 4 conv + 3 linear, 36.8K params.
Features: Ne, Nmu, Age, Ze (4 of the 6 available features).
Expected fraction error: ~0.107
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

from load_data import load_train, load_test
from verify import evaluate

DEVICE = "cuda"
BATCH_SIZE = 1024
MAX_EPOCHS = 80
LR = 3e-4
SEED = 42

def p(msg): print(msg, flush=True)


class CompositionDataset(Dataset):
    def __init__(self, matrices, features, labels, augment=False):
        self.matrices = matrices
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        mat = torch.from_numpy(self.matrices[idx].copy().astype(np.float32))
        feat = self.features[idx].astype(np.float32)
        label = int(self.labels[idx])

        if self.augment:
            if torch.rand(1) < 0.5: mat = torch.rot90(mat, 1, [0, 1])
            if torch.rand(1) < 0.5: mat = torch.rot90(mat, 2, [0, 1])
            if torch.rand(1) < 0.5: mat = torch.flip(mat, dims=[0])

        # Features: [E, Ze, Az, Ne, Nmu, Age] → normalize [Ne, Nmu, Age, Ze]
        Ne = (feat[3] - 5.31) / 0.5
        Nmu = (feat[4] - 4.3) / 0.42
        Age = feat[5] - 1.0
        Ze = feat[1] / 60.0
        reco = torch.tensor([Ne, Nmu, Age, Ze], dtype=torch.float32)

        return mat, reco, label


class LeNet(nn.Module):
    def __init__(self, n_classes=5, n_reco=4, n_channels=2):
        super().__init__()
        ch = 16
        self.conv0 = nn.Conv2d(n_channels, ch, 2)
        self.conv1 = nn.Conv2d(ch, ch, 3)
        self.conv2 = nn.Conv2d(ch, ch, 4)
        self.conv3 = nn.Conv2d(ch, ch, 2)
        self.fc1 = nn.Linear(ch * 3 * 3 + n_reco, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.fc_cnn = nn.Linear(ch * 3 * 3, n_classes)

    def forward(self, x, x_reco=None, return_cnn_out=False):
        x = x.permute(0, 3, 1, 2).float()
        x = F.max_pool2d(F.relu(self.conv0(x)), 2, stride=1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x_cnn = x
        if x_reco is not None:
            x = torch.cat((x, x_reco), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        if return_cnn_out:
            return out, self.fc_cnn(x_cnn)
        return out


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()

    # Load data
    p("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()

    full_train = CompositionDataset(X_train, f_train, y_train, augment=True)
    test_ds = CompositionDataset(X_test, f_test, y_test)

    # Split train → train/val for checkpoint selection
    n = len(full_train)
    n_val = int(n * 0.23)
    n_trn = n - n_val
    trn_ds, val_indices = torch.utils.data.random_split(
        full_train, [n_trn, n_val], generator=torch.Generator().manual_seed(SEED))
    val_raw = CompositionDataset(X_train, f_train, y_train, augment=False)
    val_ds = torch.utils.data.Subset(val_raw, val_indices.indices)
    p(f"  train={n_trn}, val={n_val}, test={len(test_ds)}")

    train_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # Class weights
    _, counts = np.unique(np.array(y_train), return_counts=True)
    weights = torch.tensor(counts[0] / counts, dtype=torch.float32)

    model = LeNet().to(DEVICE)
    p(f"  Params: {sum(pp.numel() for pp in model.parameters()):,}")

    ce = nn.CrossEntropyLoss(weight=weights.to(DEVICE), reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=7)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        tc, tt = 0, 0
        for mat, reco, y in train_loader:
            mat, reco, y = mat.to(DEVICE), reco.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out, out_cnn = model(mat, reco, return_cnn_out=True)
            loss = ce(out, y).mean() / 5 + ce(out_cnn, y).mean() / 5
            pr = F.softmax(out, 1)
            ip, cp = torch.unique(pr.argmax(1), sorted=True, return_counts=True)
            yp = torch.zeros(5, device=DEVICE); yp[ip.long()] = cp.float()
            it, ct = torch.unique(y, sorted=True, return_counts=True)
            yt = torch.zeros(5, device=DEVICE); yt[it.long()] = ct.float()
            loss += ((yp/yp.sum() - yt/yt.sum())**2).mean()
            loss.backward()
            optimizer.step()
            tc += (out.argmax(1) == y).sum().item(); tt += len(y)

        model.eval()
        vl, vt = 0, 0
        with torch.no_grad():
            for mat, reco, y in val_loader:
                mat, reco, y = mat.to(DEVICE), reco.to(DEVICE), y.to(DEVICE)
                out, out_cnn = model(mat, reco, return_cnn_out=True)
                loss = ce(out, y).mean() / 5 + ce(out_cnn, y).mean() / 5
                vl += loss.item() * len(y); vt += len(y)
        val_loss = vl / vt
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
            p(f"  Ep {epoch+1}/{MAX_EPOCHS}: acc={tc/tt:.4f} val_loss={val_loss:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} [{time.time()-t0:.0f}s]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience_counter = 0
            torch.save(model.state_dict(), "model_baseline.pt")
        else:
            patience_counter += 1
            if patience_counter >= 7:
                p(f"  Early stopping at epoch {epoch+1}"); break

    # Evaluate on test — through verify.py (the ONLY official metric)
    model.load_state_dict(torch.load("model_baseline.pt", weights_only=True))
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    all_preds = []
    with torch.no_grad():
        for mat, reco, y in test_loader:
            mat, reco = mat.to(DEVICE), reco.to(DEVICE)
            out = model(mat, reco)
            all_preds.append(out.argmax(1).cpu().numpy())

    predictions = np.concatenate(all_preds)
    np.savez("predictions_baseline.npz", predictions=predictions)

    # Official evaluation
    evaluate(predictions, "Baseline: LeNet CNN (36.8K params), 4 features, published architecture",
             pred_path="predictions_baseline.npz")

    p(f"  Time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
