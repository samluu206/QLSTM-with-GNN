"""
hypernet.py — Step 3 of the GNN-Conditioned VQC pipeline.

HyperNet MLP: graph embedding e(G) → input encoding (W_in, b_in) + VQC weights ϕ₁…ϕ₆.

Each VQC uses BasicEntanglerLayers with weight shape (n_qlayers, n_qubits).
W_in encodes the QLSTM input [θ_t, y_t, h_{t-1}] into qubit angles (graph-conditioned).
All outputs are generated once per graph and reused across all T QLSTM steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HyperNet(nn.Module):
    """
    Two-hidden-layer MLP that maps a graph embedding to QLSTM weights.

    forward(e_G) returns (W_in, b_in, ϕ₁, …, ϕ₆):
      W_in : (B, n_qubits, concat_size)  — input projection matrix
      b_in : (B, n_qubits)               — input projection bias
      ϕ₁…ϕ₆ : each (B, n_qlayers, n_qubits)

    Total output: n_qubits*(concat_size+1) + n_vqcs*n_qlayers*n_qubits
                = 4*(7+1) + 6*2*4 = 32 + 48 = 80.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_vqcs: int = 6,
        n_qlayers: int = 2,
        n_qubits: int = 4,
        concat_size: int = 7,  # n_qaoa_params(2) + y_t(1) + h_dim(4)
    ) -> None:
        super().__init__()
        self.n_vqcs = n_vqcs
        self.n_qlayers = n_qlayers
        self.n_qubits = n_qubits
        self.concat_size = concat_size
        self.weights_per_vqc = n_qlayers * n_qubits  # 8

        total_out = (
            n_qubits * concat_size          # W_in: 4×7 = 28
            + n_qubits                       # b_in: 4
            + n_vqcs * self.weights_per_vqc  # phis: 6×8 = 48
        )  # = 80

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_out),
        )

    def forward(self, e_G: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            e_G: graph embedding, shape (B, embed_dim).

        Returns:
            (W_in, b_in, ϕ₁, …, ϕ₆) — 8 tensors total.
        """
        out = self.mlp(e_G)  # (B, 80)

        win_size = self.n_qubits * self.concat_size   # 28
        bin_size = self.n_qubits                      # 4
        vqc_size = self.n_vqcs * self.weights_per_vqc # 48

        W_in_flat, b_in, phi_flat = out.split([win_size, bin_size, vqc_size], dim=-1)

        W_in = W_in_flat.reshape(*W_in_flat.shape[:-1], self.n_qubits, self.concat_size)
        # W_in: (B, 4, 7)

        chunks = phi_flat.split(self.weights_per_vqc, dim=-1)
        phis = tuple(
            c.reshape(*c.shape[:-1], self.n_qlayers, self.n_qubits) for c in chunks
        )  # 6 × (B, 2, 4)

        return (W_in, b_in) + phis  # (W_in, b_in, phi_1, ..., phi_6)


# ---------------------------------------------------------------------------
# Verification demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Sanity-checks on a random graph embedding:
      1. Output is (W_in(1,4,7), b_in(1,4), ϕ₁…ϕ₆ each (1,2,4)) — 8 tensors.
      2. All 6 phi tensors are numerically distinct.
      3. Gradient flows back through the MLP to e_G.
    """
    from gnn import GNNEncoder, nx_to_pyg
    import networkx as nx

    G = nx.path_graph(6)
    gnn      = GNNEncoder(node_feature_dim=1, dim_h=8, n_qaoa_params=2, qlstm_h_dim=4)
    hypernet = HyperNet(embed_dim=64, hidden_dim=128, n_vqcs=6, n_qlayers=2,
                        n_qubits=4, concat_size=7)

    total_out = hypernet.n_qubits * hypernet.concat_size + hypernet.n_qubits \
                + hypernet.n_vqcs * hypernet.weights_per_vqc
    print(f"HyperNet params : {sum(p.numel() for p in hypernet.parameters()):,}")
    print(f"Total output    : {total_out}  "
          f"(W_in={hypernet.n_qubits*hypernet.concat_size}, "
          f"b_in={hypernet.n_qubits}, "
          f"phis={hypernet.n_vqcs}×{hypernet.weights_per_vqc})\n")

    # 1. Shape check
    gnn.eval(); hypernet.eval()
    with torch.no_grad():
        _, _, e_G = gnn(nx_to_pyg(G))     # (1, 64)
        out = hypernet(e_G)               # (W_in, b_in, phi_1,...,phi_6) — 8 tensors

    W_in, b_in, *phis = out
    assert W_in.shape == (1, 4, 7), f"W_in shape wrong: {W_in.shape}"
    assert b_in.shape == (1, 4),    f"b_in shape wrong: {b_in.shape}"
    assert len(phis) == 6,          f"Expected 6 phis, got {len(phis)}"
    for k, phi in enumerate(phis, 1):
        assert phi.shape == (1, 2, 4), f"ϕ{k} shape wrong: {phi.shape}"
    print(f"Shape check  : PASSED  — W_in(1,4,7), b_in(1,4), 6×ϕ(1,2,4)")

    # 2. All 6 phi tensors are distinct
    for i in range(6):
        for j in range(i + 1, 6):
            assert not torch.allclose(phis[i], phis[j]), \
                f"ϕ{i+1} and ϕ{j+1} are identical — MLP may be collapsed"
    print("Distinct phis: PASSED  — all 6 VQC weight tensors differ")

    # 3. Gradient flows from all outputs back through HyperNet
    gnn.train(); hypernet.train()
    _, _, e_G = gnn(nx_to_pyg(G))
    out = hypernet(e_G)
    loss = sum(t.sum() for t in out)
    loss.backward()

    for name, p in hypernet.named_parameters():
        assert p.grad is not None,         f"No gradient: {name}"
        assert p.grad.abs().sum() > 0,     f"Zero gradient: {name}"
    print("Gradient check: PASSED  — gradients reach all HyperNet params")

    print("\nHyperNet ready.")


if __name__ == "__main__":
    _demo()
