"""
gnn.py — Step 2 of the GNN-Conditioned VQC pipeline.

GNN encoder adapted from chizhang24/Qracle (DGNN architecture).
Qracle predicts VQE initial parameters from Hamiltonian graphs;
here we adapt it to predict QAOA initial parameters from MaxCut graphs.

Architecture (DGNN from Qracle, dim_h=8):
  GCNConv(1 → 16) → GCNConv(16 → 16) →
  GATConv(16 → 32) → GATConv(32 → 32) →
  GINConv(32 → 64) → global_mean_pool

Three outputs per graph:
  theta0 : initial QAOA parameters      ℝ^{n_qaoa_params}
  h0     : initial QLSTM hidden state   ℝ^{qlstm_h_dim}
  e_G    : graph embedding (→ HyperNet) ℝ^{dim_h*8} = ℝ^{64}
"""

from __future__ import annotations

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool


def nx_to_pyg(graph: nx.Graph) -> Data:
    """NetworkX graph → PyG Data with normalised-degree node features.

    Node features x: shape (n, 1), each entry = degree / max_degree.
    Edge index: both (u→v) and (v→u) so the graph is undirected in PyG.
    """
    g = nx.convert_node_labels_to_integers(graph)

    degrees = torch.tensor(
        [d for _, d in g.degree()], dtype=torch.float
    ).unsqueeze(1)                              # (n, 1)
    x = degrees / degrees.max().clamp(min=1.0)  # normalise to [0, 1]

    if g.number_of_edges() > 0:
        src, dst = zip(*g.edges())
        edge_index = torch.tensor(
            [list(src) + list(dst), list(dst) + list(src)], dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


class GNNEncoder(nn.Module):
    """
    DGNN architecture adapted from chizhang24/Qracle.

    Channel widths scale with dim_h (same convention as Qracle):
      GCN channels : dim_h * 2
      GAT channels : dim_h * 4
      GIN channels : dim_h * 8  ← becomes the e_G embedding dimension

    With default dim_h=8: GCN=16, GAT=32, GIN=64.

    forward() returns batched tensors (B, *).
    Single-graph usage: call .squeeze(0) on each output, or pass a Data
    object without a batch attribute (batch is inferred as all-zeros).
    """

    def __init__(
        self,
        node_feature_dim: int = 1,
        dim_h: int = 8,
        n_qaoa_params: int = 2,
        qlstm_h_dim: int = 4,
    ) -> None:
        super().__init__()

        ch_gcn = dim_h * 2   # 16
        ch_gat = ch_gcn * 2  # 32
        ch_gin = ch_gat * 2  # 64  ← e_G dimension

        # Two GCNConv layers (from Qracle DGNN)
        self.conv1 = GCNConv(node_feature_dim, ch_gcn)
        self.conv2 = GCNConv(ch_gcn, ch_gcn)

        # Two GATConv layers (from Qracle DGNN)
        self.gat_conv1 = GATConv(ch_gcn, ch_gat)
        self.gat_conv2 = GATConv(ch_gat, ch_gat)

        # GINConv layer (from Qracle DGNN)
        gin_mlp = nn.Sequential(
            nn.Linear(ch_gat, ch_gin),
            nn.ReLU(),
            nn.Linear(ch_gin, ch_gin),
        )
        self.gin_conv1 = GINConv(gin_mlp)

        # Output heads — replace Qracle's single self.out with three outputs
        self.theta0_head = nn.Linear(ch_gin, n_qaoa_params)
        self.h0_head     = nn.Linear(ch_gin, qlstm_h_dim)
        # e_G is the raw global_mean_pool output, shape (B, ch_gin)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            data: PyG Data with fields x, edge_index, and optionally batch.

        Returns:
            theta0 : (B, n_qaoa_params)  initial QAOA parameters
            h0     : (B, qlstm_h_dim)    initial QLSTM hidden state
            e_G    : (B, ch_gin)         graph embedding for HyperNet
        """
        x, edge_index = data.x, data.edge_index
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        # Forward pass identical to Qracle DGNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.gat_conv1(x, edge_index))
        x = F.relu(self.gat_conv2(x, edge_index))
        x = F.relu(self.gin_conv1(x, edge_index))
        e_G = global_mean_pool(x, batch)          # (B, ch_gin)

        return self.theta0_head(e_G), self.h0_head(e_G), e_G


# ---------------------------------------------------------------------------
# Verification demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Sanity-checks on a 3-regular 6-node graph:
      1. Correct output shapes: (1, 2), (1, 4), (1, 64).
      2. Gradient flow reaches every parameter.
    """
    G = nx.random_regular_graph(3, 6, seed=42)
    data = nx_to_pyg(G)

    model = GNNEncoder(node_feature_dim=1, dim_h=8, n_qaoa_params=2, qlstm_h_dim=4)

    print(f"Graph : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"x     : {tuple(data.x.shape)}")
    print(f"edges : {tuple(data.edge_index.shape)}")

    # 1. Shape check (eval, no grad)
    model.eval()
    with torch.no_grad():
        theta0, h0, e_G = model(data)

    print(f"\ntheta0 : {tuple(theta0.shape)}  (expected (1, 2))")
    print(f"h0     : {tuple(h0.shape)}  (expected (1, 4))")
    print(f"e_G    : {tuple(e_G.shape)}  (expected (1, 64))")

    assert theta0.shape == (1, 2),  f"theta0 shape wrong: {theta0.shape}"
    assert h0.shape     == (1, 4),  f"h0 shape wrong: {h0.shape}"
    assert e_G.shape    == (1, 64), f"e_G shape wrong: {e_G.shape}"
    print("Shape check  : PASSED")

    # 2. Gradient flow through all parameters
    model.train()
    theta0, h0, e_G = model(data)
    (theta0.sum() + h0.sum() + e_G.sum()).backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"
    print("Gradient check: PASSED")

    print("\nGNNEncoder ready.")


if __name__ == "__main__":
    _demo()
