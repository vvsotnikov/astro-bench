"""v39: Extract CNN features (penultimate layer) and train GBM on top.
Uses v8 model to extract 256-dim CNN embeddings, combines with engineered
features, then trains HistGradientBoosting on the concatenation.
GBM may produce better-calibrated predictions than the CNN head.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.ensemble import HistGradientBoostingClassifier
import subprocess, gc, time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DEVICE = "cuda"
BATCH_SIZE = 4096

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    return np.stack([
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu, (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
    ], axis=1).astype(np.float32)


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class HybridModel(nn.Module):
    def __init__(self, n_feat=13, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )
    def forward(self, mat, feat):
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))

    def extract_features(self, mat, feat):
        """Extract penultimate features (before final linear)."""
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        # Pass through head up to last ReLU
        x = self.head[0](combined)  # Linear 512
        x = self.head[1](x)  # BN
        x = self.head[2](x)  # ReLU
        x = self.head[3](x)  # Dropout
        x = self.head[4](x)  # Linear 256
        x = self.head[5](x)  # BN
        x = self.head[6](x)  # ReLU
        return x  # 256-dim features


def extract_cnn_features(model, mat_data, feat_data, batch_size=BATCH_SIZE):
    """Extract features for all data."""
    model.eval()
    features_list = []
    with torch.no_grad():
        for i in range(0, len(feat_data), batch_size):
            end = min(i + batch_size, len(feat_data))
            mb = mat_data[i:end].to(DEVICE)
            fb = feat_data[i:end].to(DEVICE)
            with autocast(device_type='cuda'):
                feats = model.extract_features(mb, fb)
            features_list.append(feats.float().cpu().numpy())
            if i % (batch_size * 50) == 0:
                p(f"  extract: {i}/{len(feat_data)}")
    return np.concatenate(features_list)


def main():
    np.random.seed(42)

    # Load v8 model
    p("Loading v8 model...")
    model = HybridModel(n_feat=13).to(DEVICE)
    state = torch.load(f"{OUT_DIR}/model_v8.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Load train data (subsample for GBM -- 5.5M is too much)
    p("Loading train data (subsample for GBM)...")
    raw_mat = np.load(f"{DATA_DIR}/composition_train/matrices.npy", mmap_mode='r')
    raw_feat = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')
    n_train = len(labels)

    # Use 2M samples for GBM training (memory/time constraint)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_train, size=2000000, replace=False)
    sample_idx.sort()

    CHUNK = 200000
    mat_chunks = []
    feat_chunks = []
    label_list = []
    for i in range(0, len(sample_idx), CHUNK):
        end = min(i + CHUNK, len(sample_idx))
        idx = sample_idx[i:end]
        p(f"  train chunk: {i}/{len(sample_idx)}")
        m = np.array(raw_mat[idx], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_chunks.append(torch.from_numpy(m))
        feat_chunks.append(engineer_features(np.array(raw_feat[idx], dtype=np.float32)))
        label_list.append(np.array(labels[idx], dtype=np.int64))

    mat_train = torch.cat(mat_chunks)
    eng_feat_train = np.concatenate(feat_chunks)
    y_train = np.concatenate(label_list)
    del mat_chunks, feat_chunks, label_list; gc.collect()

    # Normalize features (same as v8 training -- but we need the stats)
    feat_mean = eng_feat_train.mean(0)
    feat_std = eng_feat_train.std(0) + 1e-6
    eng_feat_norm = (eng_feat_train - feat_mean) / feat_std
    feat_train_t = torch.from_numpy(eng_feat_norm)

    p("Extracting CNN features for train...")
    t0 = time.time()
    cnn_feats_train = extract_cnn_features(model, mat_train, feat_train_t)
    p(f"  CNN features: {cnn_feats_train.shape}, {time.time()-t0:.0f}s")

    # Combine CNN features with engineered features for GBM
    X_train_gbm = np.concatenate([cnn_feats_train, eng_feat_train], axis=1)
    p(f"  GBM input: {X_train_gbm.shape}")
    del mat_train, feat_train_t, cnn_feats_train; gc.collect()

    # Load test
    p("Loading test data...")
    raw_mat_test = np.load(f"{DATA_DIR}/composition_test/matrices.npy", mmap_mode='r')
    raw_feat_test = np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r')
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)

    mat_list = []
    for i in range(0, len(y_test), 250000):
        end = min(i + 250000, len(y_test))
        m = np.array(raw_mat_test[i:end], dtype=np.float32)
        m = np.log1p(m).transpose(0, 3, 1, 2)
        mat_list.append(torch.from_numpy(m))
    mat_test = torch.cat(mat_list)
    del mat_list; gc.collect()

    test_eng_feats = engineer_features(np.array(raw_feat_test[:], dtype=np.float32))
    test_eng_norm = (test_eng_feats - feat_mean) / feat_std
    feat_test_t = torch.from_numpy(test_eng_norm)

    p("Extracting CNN features for test...")
    cnn_feats_test = extract_cnn_features(model, mat_test, feat_test_t)

    X_test_gbm = np.concatenate([cnn_feats_test, test_eng_feats], axis=1)
    p(f"  GBM test input: {X_test_gbm.shape}")
    del mat_test, feat_test_t, cnn_feats_test; gc.collect()

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Train GBM
    for n_iter in [500, 1000]:
        for max_depth in [6, 8]:
            for lr_gbm in [0.05, 0.1]:
                config_name = f"iter{n_iter}_d{max_depth}_lr{lr_gbm}"
                p(f"\nGBM config: {config_name}")
                t0 = time.time()

                gbm = HistGradientBoostingClassifier(
                    max_iter=n_iter,
                    max_depth=max_depth,
                    learning_rate=lr_gbm,
                    min_samples_leaf=50,
                    l2_regularization=0.01,
                    random_state=42,
                    verbose=0,
                )
                gbm.fit(X_train_gbm, y_train)
                elapsed = time.time() - t0

                preds = gbm.predict(X_test_gbm)
                test_acc = (preds == y_test).mean()

                # Get probabilities
                probs = gbm.predict_proba(X_test_gbm)

                # Verify
                np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))
                result = subprocess.run(
                    ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
                    capture_output=True, text=True,
                    cwd="/home/vladimir/cursor_projects/astro-agents"
                )
                frac_err = 1.0
                for line in result.stdout.split('\n'):
                    if 'mean fraction error' in line.lower():
                        try:
                            frac_err = float(line.split(':')[-1].strip())
                        except:
                            pass
                        break

                p(f"  acc={test_acc:.4f} frac_err={frac_err:.4f} time={elapsed:.0f}s")

                # Save probs for ensembling
                np.save(f"{OUT_DIR}/probs_v39_{config_name}.npy", probs)

    p("---")
    p("description: CNN feature extraction + GBM, various configs")


if __name__ == "__main__":
    main()
