"""Graph Neural Network: treat active detector pixels as nodes, spatial adjacency as edges."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class GammaDataset(Dataset):
    def __init__(self, split: str, mean=None, std=None):
        self.matrices = np.load(f"data/gamma_{split}/matrices.npy", mmap_mode="r")
        self.features = np.load(f"data/gamma_{split}/features.npy", mmap_mode="r")
        self.labels = np.load(f"data/gamma_{split}/labels_gamma.npy", mmap_mode="r")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)  # [16, 16, 2]
        mat = np.transpose(mat, (2, 0, 1))  # [2, 16, 16]

        feat = self.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        Ne_minus_Nmu = Ne - Nmu
        cos_Ze = np.cos(np.deg2rad(Ze))
        sin_Ze = np.sin(np.deg2rad(Ze))
        all_feats = np.array([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze], dtype=np.float32)

        if self.mean is not None:
            mat_flat = mat.flatten()
            mat_flat = (mat_flat - self.mean[:512]) / (self.std[:512] + 1e-8)
            mat = mat_flat.reshape(mat.shape)
            all_feats = (all_feats - self.mean[512:]) / (self.std[512:] + 1e-8)

        label = int(self.labels[idx])

        # Convert to graph: active pixels as nodes
        # mat shape: [2, 16, 16] -> flatten to [2, 256]
        node_features = mat.reshape(2, -1).T  # [256, 2]

        # Keep only active nodes (where sum of both channels > small threshold)
        activity = node_features.sum(axis=1)
        active_mask = activity > 1e-6
        active_nodes = node_features[active_mask]  # [n_active, 2]

        if len(active_nodes) == 0:
            # Fallback: use at least center node
            active_nodes = np.array([[mat[0, 8, 8], mat[1, 8, 8]]], dtype=np.float32)
            active_idx = np.array([136])  # center
        else:
            active_idx = np.where(active_mask)[0]

        # Build edges: 4-connected adjacency on original grid
        edges = []
        for i in range(len(active_idx)):
            pos_i = active_idx[i]
            x_i, y_i = pos_i // 16, pos_i % 16

            for j in range(i + 1, len(active_idx)):
                pos_j = active_idx[j]
                x_j, y_j = pos_j // 16, pos_j % 16

                # 4-connected neighbors
                if (abs(x_i - x_j) == 1 and y_i == y_j) or (abs(y_i - y_j) == 1 and x_i == x_j):
                    edges.append([i, j])
                    edges.append([j, i])

        if len(edges) == 0:
            # No edges: add self-loops for isolated nodes
            edges = [[i, i] for i in range(len(active_nodes))]

        edges = np.array(edges, dtype=np.int64).T
        if edges.size > 0:
            edge_index = torch.from_numpy(edges)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        x = torch.from_numpy(active_nodes)
        return x, edge_index, torch.from_numpy(all_feats), label


def compute_stats(dataset):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(100_000, len(dataset)), replace=False)

    mats, feats = [], []
    for idx in indices:
        m = dataset.matrices[idx].astype(np.float32).transpose(2, 0, 1).flatten()
        f = dataset.features[idx].astype(np.float32)
        E, Ze, Az, Ne, Nmu = f
        f = np.array([E, Ze, Az, Ne, Nmu, Ne-Nmu, np.cos(np.deg2rad(Ze)), np.sin(np.deg2rad(Ze))], dtype=np.float32)
        mats.append(m)
        feats.append(f)
    mats, feats = np.stack(mats), np.stack(feats)
    mean = np.concatenate([mats.mean(0), feats.mean(0)])
    std = np.concatenate([mats.std(0), feats.std(0)])
    std[std == 0] = 1.0
    return mean, std


class GNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(2, 64)
        self.gcn2 = GCNConv(64, 128)

        self.feat_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch, feat):
        x = torch.relu(self.gcn1(x, edge_index))
        x = torch.relu(self.gcn2(x, edge_index))
        x = global_mean_pool(x, batch)

        x_feat = self.feat_mlp(feat)
        combined = torch.cat([x, x_feat], dim=1)
        return self.fusion(combined).squeeze(-1)


device = torch.device("cuda:0")
print(f"Device: {device}\n")

raw_train = GammaDataset("train")
mean, std = compute_stats(raw_train)
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

def compute_survival_75(scores):
    is_gamma = test_labels == 0
    is_hadron = test_labels == 1
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

# Custom collate for graphs
def collate_fn(batch):
    from torch_geometric.data import Batch
    graphs = []
    feats = []
    labels = []

    for x, edge_index, feat, label in batch:
        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)
        feats.append(feat)
        labels.append(label)

    batch_graph = Batch.from_data_list(graphs)
    feats = torch.stack(feats)
    labels = torch.tensor(labels, dtype=torch.long)

    return batch_graph, feats, labels

print("Preparing data...")
n_train = int(0.8 * len(raw_train))
n_val = len(raw_train) - n_train

train_idx, val_idx = np.arange(n_train), np.arange(n_train, len(raw_train))
train_ds = [GammaDataset("train", mean=mean, std=std)[i] for i in train_idx]
val_ds = [GammaDataset("train", mean=mean, std=std)[i] for i in val_idx]
test_ds = [GammaDataset("test", mean=mean, std=std)[i] for i in range(len(GammaDataset("test")))]

from torch.utils.data import Subset
train_subset = Subset(GammaDataset("train", mean=mean, std=std), train_idx)
val_subset = Subset(GammaDataset("train", mean=mean, std=std), val_idx)
test_subset = Subset(GammaDataset("test", mean=mean, std=std), np.arange(len(GammaDataset("test"))))

train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn)
test_loader = DataLoader(test_subset, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn)

print("Training GNN...")
model = GNNModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_survival = 1.0
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_graph, feat, y in train_loader:
        batch_graph = batch_graph.to(device)
        feat, y = feat.to(device), y.float().to(device)
        scores = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch, feat)
        loss = criterion(scores, (y == 0).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            scores = []
            for batch_graph, feat, _ in val_loader:
                batch_graph = batch_graph.to(device)
                feat = feat.to(device)
                scores.append(model(batch_graph.x, batch_graph.edge_index, batch_graph.batch, feat).cpu().numpy())
        val_survival = compute_survival_75(np.concatenate(scores))
        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, val_survival={val_survival:.4e}")

        if val_survival < best_survival:
            best_survival = val_survival
            torch.save(model.state_dict(), "/tmp/model_gnn.pt")

model.load_state_dict(torch.load("/tmp/model_gnn.pt"))
model.eval()

test_scores = []
with torch.no_grad():
    for batch_graph, feat, _ in test_loader:
        batch_graph = batch_graph.to(device)
        feat = feat.to(device)
        scores = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch, feat).cpu().numpy()
        test_scores.append(scores)

test_scores = np.concatenate(test_scores)
test_survival = compute_survival_75(test_scores)

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v52.npz", gamma_scores=test_scores)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: Graph Neural Network on sparse detector (active pixels as nodes, spatial edges)")
