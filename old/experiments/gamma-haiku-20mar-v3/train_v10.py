"""v10: 3-model ensemble (v1-like + v3 features + v5 RF)

Ensemble best models from v1-v9.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from load_data import load_train, load_test
from verify import evaluate


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels):
        self.matrices = torch.from_numpy(matrices.astype(np.float32)).permute(0, 3, 1, 2)
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.matrices[idx], self.features[idx], self.labels[idx]


def extract_features(matrices, features):
    """Extract muon statistics for RF."""
    muon_channel = matrices[:, :, :, 1]
    electron_channel = matrices[:, :, :, 0]

    muon_stats = np.stack([
        muon_channel.reshape(len(matrices), -1).mean(axis=1),
        muon_channel.reshape(len(matrices), -1).max(axis=1),
        muon_channel.reshape(len(matrices), -1).sum(axis=1),
        muon_channel.reshape(len(matrices), -1).std(axis=1),
    ], axis=1)

    electron_stats = np.stack([
        electron_channel.reshape(len(matrices), -1).mean(axis=1),
        electron_channel.reshape(len(matrices), -1).max(axis=1),
    ], axis=1)

    return np.concatenate([features, muon_stats, electron_stats], axis=1)


class MuonCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.muon_path = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.electron_path = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 * 16 + 16 * 16 + 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, matrices, features):
        muon = matrices[:, 1:2, :, :]
        electron = matrices[:, 0:1, :, :]
        muon_feat = self.muon_path(muon).flatten(1)
        electron_feat = self.electron_path(electron).flatten(1)
        combined = torch.cat([muon_feat, electron_feat, features], dim=1)
        return self.fusion(combined)


def main():
    device = torch.device("cuda:0")
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()

    # 1. CNN predictions (if model exists, load and predict)
    print("Getting CNN predictions...")
    try:
        test_ds = GammaDataset(X_test, f_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

        model = MuonCNN().to(device)
        model.load_state_dict(torch.load("model_v1.pt"))
        model.eval()

        cnn_scores = []
        with torch.no_grad():
            for matrices, features, _ in test_loader:
                matrices, features = matrices.to(device), features.to(device)
                logits = model(matrices, features)
                cnn_scores.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())
        cnn_scores = np.concatenate(cnn_scores)
        print(f"  CNN scores: {len(cnn_scores)}")
    except Exception as e:
        print(f"  CNN failed: {e}, skipping")
        cnn_scores = None

    # 2. RF on engineered features
    print("Training RandomForest...")
    X_train_feat = extract_features(X_train, f_train)
    X_test_feat = extract_features(X_test, f_test)

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)

    class_weight = {
        0: len(y_train) / (2 * (y_train == 0).sum()),
        1: len(y_train) / (2 * (y_train == 1).sum()),
    }

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=30, n_jobs=-1, random_state=42
    )
    rf_model.fit(X_train_feat, y_train)
    rf_probs = rf_model.predict_proba(X_test_feat)
    rf_scores = rf_probs[:, 0]
    print(f"  RF scores: {len(rf_scores)}")

    # Ensemble
    if cnn_scores is not None:
        ensemble_scores = 0.6 * cnn_scores + 0.4 * rf_scores
        desc = "v10: Ensemble (60% CNN + 40% RF on engineered features)"
    else:
        ensemble_scores = rf_scores
        desc = "v10: RandomForest (CNN unavailable)"

    evaluate(ensemble_scores, desc)


if __name__ == "__main__":
    main()
