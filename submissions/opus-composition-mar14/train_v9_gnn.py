"""v9: Graph Neural Network on sparse detector grid.
Instead of treating the 16x16 grid as a dense image (85% zeros),
treat non-zero detector stations as nodes in a graph.

Each node has features:
- Position (x, y)
- Electron/photon density
- Muon density
- Distance from center/max
Connected by edges based on spatial proximity (k-nearest neighbors).

This exploits the sparse spatial structure directly.
Implemented with pure PyTorch (no torch_geometric needed).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
import gc

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-composition-mar14"
DEVICE = "cuda"
BATCH_SIZE = 2048  # Smaller due to variable graph sizes
EPOCHS = 20
LR = 1e-3
LABEL_SMOOTH = 0.05
SEED = 42
K_NEIGHBORS = 6  # edges per node
MAX_NODES = 64   # max non-zero cells to keep (pad/truncate)

def p(msg):
    print(msg, flush=True)

def engineer_features(f):
    E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    feats = [
        E, Ze, Ne, Nmu,
        np.sin(np.radians(Ze)), np.cos(np.radians(Ze)),
        np.sin(np.radians(Az)), np.cos(np.radians(Az)),
        Ne - Nmu, Ne + Nmu,
        (Ne - Nmu) / (Ne + Nmu + 1e-6),
        Ne - E, Nmu - E,
    ]
    return np.stack(feats, axis=1).astype(np.float32)


class MessagePassingLayer(nn.Module):
    """Simple message passing: aggregate neighbor features and update."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 2, out_dim),  # +2 for relative position
            nn.ReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x, pos, edge_idx, mask):
        """
        x: (B, N, D) node features
        pos: (B, N, 2) positions
        edge_idx: (B, N, K) neighbor indices
        mask: (B, N) valid node mask
        """
        B, N, D = x.shape
        K = edge_idx.shape[2]

        # Gather neighbor features
        edge_idx_flat = edge_idx.reshape(B, -1)  # (B, N*K)
        neighbors = torch.gather(x, 1, edge_idx_flat.unsqueeze(-1).expand(-1, -1, D))  # (B, N*K, D)
        neighbors = neighbors.reshape(B, N, K, D)

        # Relative positions
        pos_expanded = pos.unsqueeze(2).expand(-1, -1, K, -1)  # (B, N, K, 2)
        neighbor_pos = torch.gather(pos, 1, edge_idx_flat.unsqueeze(-1).expand(-1, -1, 2))
        neighbor_pos = neighbor_pos.reshape(B, N, K, 2)
        rel_pos = neighbor_pos - pos_expanded

        # Edge features: [node, neighbor, rel_pos]
        x_expanded = x.unsqueeze(2).expand(-1, -1, K, -1)  # (B, N, K, D)
        edge_feat = torch.cat([x_expanded, neighbors, rel_pos], dim=-1)  # (B, N, K, 2D+2)
        messages = self.edge_mlp(edge_feat)  # (B, N, K, out_dim)

        # Aggregate (mean)
        agg = messages.mean(dim=2)  # (B, N, out_dim)

        # Update
        updated = self.node_mlp(torch.cat([x, agg], dim=-1))  # (B, N, out_dim)

        # Mask invalid nodes
        updated = updated * mask.unsqueeze(-1)
        return updated


class GNNClassifier(nn.Module):
    def __init__(self, node_dim=4, n_scalar=13, hidden_dim=64, n_classes=5):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mp1 = MessagePassingLayer(hidden_dim, hidden_dim)
        self.mp2 = MessagePassingLayer(hidden_dim, hidden_dim)
        self.mp3 = MessagePassingLayer(hidden_dim, hidden_dim)

        # Global readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Scalar feature branch
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalar, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, node_feat, pos, edge_idx, mask, scalar_feat):
        x = self.node_embed(node_feat)
        x = self.mp1(x, pos, edge_idx, mask)
        x = self.mp2(x, pos, edge_idx, mask)
        x = self.mp3(x, pos, edge_idx, mask)

        # Global mean pooling (masked)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        graph_feat = (x * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        graph_feat = self.readout(graph_feat)

        # Combine with scalar features
        scalar_out = self.scalar_mlp(scalar_feat)
        combined = torch.cat([graph_feat, scalar_out], dim=-1)
        return self.head(combined)


def matrix_to_graph(matrices_np, k=K_NEIGHBORS, max_nodes=MAX_NODES):
    """Convert dense 16x16x2 matrices to graph representation.
    Returns: node_features (N, max_nodes, 4), positions (N, max_nodes, 2),
             edge_indices (N, max_nodes, K), masks (N, max_nodes)
    """
    n = len(matrices_np)
    node_features = np.zeros((n, max_nodes, 4), dtype=np.float32)  # [e_density, mu_density, log1p_e, log1p_mu]
    positions = np.zeros((n, max_nodes, 2), dtype=np.float32)
    edge_indices = np.zeros((n, max_nodes, k), dtype=np.int64)
    masks = np.zeros((n, max_nodes), dtype=np.float32)

    for i in range(n):
        mat = matrices_np[i]  # (16, 16, 2)
        ch0 = mat[:, :, 0]  # electron
        ch1 = mat[:, :, 1]  # muon

        # Find non-zero cells
        nonzero = (ch0 > 0) | (ch1 > 0)
        ys, xs = np.where(nonzero)
        n_nodes = len(ys)

        if n_nodes == 0:
            continue

        # Truncate if needed (keep highest-energy cells)
        if n_nodes > max_nodes:
            total_energy = ch0[nonzero] + ch1[nonzero]
            top_idx = np.argsort(total_energy)[-max_nodes:]
            ys = ys[top_idx]
            xs = xs[top_idx]
            n_nodes = max_nodes

        # Node features
        for j in range(n_nodes):
            y, x = ys[j], xs[j]
            node_features[i, j] = [ch0[y, x], ch1[y, x], np.log1p(ch0[y, x]), np.log1p(ch1[y, x])]
            positions[i, j] = [x / 15.0, y / 15.0]  # normalized positions
            masks[i, j] = 1.0

        # Build k-nearest neighbor edges
        if n_nodes > 1:
            pos = np.column_stack([xs, ys]).astype(float)
            for j in range(n_nodes):
                dists = np.sqrt(((pos - pos[j]) ** 2).sum(axis=1))
                dists[j] = 1e10  # exclude self
                k_actual = min(k, n_nodes - 1)
                neighbors = np.argsort(dists)[:k_actual]
                edge_indices[i, j, :k_actual] = neighbors
                # Pad remaining edges with self-loop
                edge_indices[i, j, k_actual:] = j
        else:
            edge_indices[i, 0, :] = 0  # self-loops

    return node_features, positions, edge_indices, masks


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, node_feat, pos, edge_idx, mask, scalar_feat, labels):
        self.node_feat = node_feat
        self.pos = pos
        self.edge_idx = edge_idx
        self.mask = mask
        self.scalar_feat = scalar_feat
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.node_feat[idx], self.pos[idx], self.edge_idx[idx],
                self.mask[idx], self.scalar_feat[idx], self.labels[idx])


def load_graph_data(split, feat_stats=None):
    p(f"Loading {split} data...")
    matrices = np.load(f"{DATA_DIR}/composition_{split}/matrices.npy", mmap_mode='r')
    raw_feats = np.load(f"{DATA_DIR}/composition_{split}/features.npy", mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/composition_{split}/labels_composition.npy", mmap_mode='r')
    n = len(labels)

    # For speed, subsample train to 2M (GNN is slower)
    if split == "train":
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n, min(2000000, n), replace=False)
        idx.sort()
        p(f"  Subsampling to {len(idx)} events")
    else:
        idx = np.arange(n)

    # Convert matrices to graph in chunks
    chunk = 50000
    nf_list, pos_list, ei_list, mask_list = [], [], [], []
    for i in range(0, len(idx), chunk):
        end = min(i + chunk, len(idx))
        batch_idx = idx[i:end]
        m = np.array(matrices[batch_idx], dtype=np.float32)
        nf, pos, ei, mask = matrix_to_graph(m)
        nf_list.append(nf)
        pos_list.append(pos)
        ei_list.append(ei)
        mask_list.append(mask)
        p(f"  {split} graph: {end}/{len(idx)}")

    node_feat = np.concatenate(nf_list)
    positions = np.concatenate(pos_list)
    edge_idx = np.concatenate(ei_list)
    masks = np.concatenate(mask_list)

    # Scalar features
    feat_chunks = []
    for i in range(0, len(idx), 500000):
        end = min(i + 500000, len(idx))
        f = np.array(raw_feats[idx[i:end]], dtype=np.float32)
        feat_chunks.append(engineer_features(f))
    scalar_feats = np.concatenate(feat_chunks)

    if feat_stats is None:
        feat_mean = scalar_feats.mean(0)
        feat_std = scalar_feats.std(0) + 1e-6
    else:
        feat_mean, feat_std = feat_stats
    scalar_feats = (scalar_feats - feat_mean) / feat_std

    y = np.array(labels[idx], dtype=np.int64)

    p(f"  {split}: n={len(y)}, node_feat={node_feat.shape}")
    return (
        torch.from_numpy(node_feat),
        torch.from_numpy(positions),
        torch.from_numpy(edge_idx),
        torch.from_numpy(masks),
        torch.from_numpy(scalar_feats),
        torch.from_numpy(y),
        (feat_mean, feat_std),
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    nf_train, pos_train, ei_train, mask_train, sf_train, y_train, stats = load_graph_data("train")
    nf_test, pos_test, ei_test, mask_test, sf_test, y_test, _ = load_graph_data("test", feat_stats=stats[0:2])

    train_ds = GraphDataset(nf_train, pos_train, ei_train, mask_train, sf_train, y_train)
    test_ds = GraphDataset(nf_test, pos_test, ei_test, mask_test, sf_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = GNNClassifier(node_dim=4, n_scalar=sf_train.shape[1], hidden_dim=64).to(DEVICE)
    p(f"Params: {sum(pp.numel() for pp in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    best_preds = None
    best_probs = None

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for nf_b, pos_b, ei_b, mask_b, sf_b, label_b in train_loader:
            nf_b = nf_b.to(DEVICE)
            pos_b = pos_b.to(DEVICE)
            ei_b = ei_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            sf_b = sf_b.to(DEVICE)
            label_b = label_b.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                out = model(nf_b, pos_b, ei_b, mask_b, sf_b)
                loss = criterion(out, label_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (out.argmax(1) == label_b).sum().item()
            total += len(label_b)
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_preds, all_probs = [], []
        tc, tt = 0, 0
        with torch.no_grad():
            for nf_b, pos_b, ei_b, mask_b, sf_b, label_b in test_loader:
                nf_b = nf_b.to(DEVICE)
                pos_b = pos_b.to(DEVICE)
                ei_b = ei_b.to(DEVICE)
                mask_b = mask_b.to(DEVICE)
                sf_b = sf_b.to(DEVICE)
                label_b = label_b.to(DEVICE)
                with autocast(device_type='cuda'):
                    out = model(nf_b, pos_b, ei_b, mask_b, sf_b)
                all_preds.append(out.argmax(1).cpu().numpy())
                all_probs.append(torch.softmax(out.float(), 1).cpu().numpy())
                tc += (out.argmax(1) == label_b).sum().item()
                tt += len(label_b)
        test_acc = tc / tt
        p(f"Ep {epoch+1}/{EPOCHS}: train={train_acc:.4f} test={test_acc:.4f} [{time.time()-t0:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(all_preds)
            best_probs = np.concatenate(all_probs)
            p(f"  >>> Best: {best_acc:.4f}")

    np.savez(f"{OUT_DIR}/predictions_v9.npz", predictions=best_preds.astype(np.int8))
    np.save(f"{OUT_DIR}/probs_v9.npy", best_probs)

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m. Best acc: {best_acc:.4f}")
    p("---")
    p(f"metric: {best_acc:.4f}")
    p(f"description: GNN (3 MP layers) + MLP, 2M train, seed={SEED}")


if __name__ == "__main__":
    main()
