"""
classical_lstm_cell.py — Classical LSTM cell (no VQCs, no HyperNet).

Drop-in replacement for QLSTMCell in the GNN-CLSTM baseline:
  - Same recurrent structure (forget/input/update/output/hidden/theta gates)
  - Gates implemented with nn.Linear instead of VQCs
  - Has its OWN learnable parameters (W_enc, W_f, W_i, W_g, W_o, W_h, W_theta)
  - Does NOT receive W_in/b_in/phis from HyperNet — clean interface

Pipeline: GNN(G) → θ₀, h₀ → ClassicalLSTMCell (no HyperNet involved)
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalLSTMCell(nn.Module):
    """
    Classical LSTM cell with clean interface (no HyperNet dependency).

    Args:
        theta_size  : dimension of θ (= 2*p for QAOA p layers; default 2)
        hidden_size : dimension of hidden state h and cell state C (default 4)

    Input concat: [θ_t(theta_size), y_t(1), h_{t-1}(hidden_size)]
                  → concat_size = theta_size + 1 + hidden_size
    """

    def __init__(self, theta_size: int = 2, hidden_size: int = 4) -> None:
        super().__init__()
        self.theta_size  = theta_size
        self.hidden_size = hidden_size
        concat_size      = theta_size + 1 + hidden_size  # 7 by default

        self.W_enc   = nn.Linear(concat_size, hidden_size)  # input projection
        self.W_f     = nn.Linear(hidden_size, hidden_size)  # forget gate
        self.W_i     = nn.Linear(hidden_size, hidden_size)  # input gate
        self.W_g     = nn.Linear(hidden_size, hidden_size)  # cell candidate
        self.W_o     = nn.Linear(hidden_size, hidden_size)  # output gate
        self.W_h     = nn.Linear(hidden_size, hidden_size)  # hidden readout
        self.W_theta = nn.Linear(hidden_size, theta_size)   # theta head (flexible)

    def forward(
        self,
        theta_t: torch.Tensor,   # (theta_size,)
        y_t:     torch.Tensor,   # scalar or (1,)
        h_prev:  torch.Tensor,   # (hidden_size,)
        C_prev:  torch.Tensor,   # (hidden_size,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Normalize dtypes: PennyLane may return float64; W_enc.weight determines target
        dtype   = next(self.parameters()).dtype
        theta_t = theta_t.to(dtype)
        y_t     = y_t.to(dtype)
        h_prev  = h_prev.to(dtype)
        C_prev  = C_prev.to(dtype)

        v     = torch.cat([theta_t, y_t.reshape(1), h_prev])  # (concat_size,)
        y_enc = self.W_enc(v)                                  # (hidden_size,)

        f_t = torch.sigmoid(self.W_f(y_enc))
        i_t = torch.sigmoid(self.W_i(y_enc))
        g_t = torch.tanh(self.W_g(y_enc))
        o_t = torch.sigmoid(self.W_o(y_enc))

        C_t = f_t * C_prev + i_t * g_t               # (hidden_size,)
        h_t = o_t * torch.tanh(self.W_h(y_enc))      # (hidden_size,) ∈ (-1,1)

        theta_next = self.W_theta(y_enc)              # (theta_size,)

        return theta_next, h_t, C_t


# ---------------------------------------------------------------------------
# Verification demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import networkx as nx
    from gnn import GNNEncoder, nx_to_pyg

    G = nx.path_graph(6)
    gnn  = GNNEncoder(node_feature_dim=1, dim_h=8, n_qaoa_params=2, qlstm_h_dim=4)
    cell = ClassicalLSTMCell(theta_size=2, hidden_size=4)

    n_params = sum(p.numel() for p in cell.parameters())
    print(f"ClassicalLSTMCell learnable params: {n_params}")
    print(f"  (theta_size=2, hidden_size=4, concat_size=7)")

    theta0, h0, _ = gnn(nx_to_pyg(G))
    theta_t = theta0.squeeze(0)
    h_prev  = h0.squeeze(0)
    C_prev  = torch.zeros(4)
    y_t     = torch.tensor(0.5)

    # 1. Shape check
    with torch.no_grad():
        theta_next, h_t, C_t = cell(theta_t, y_t, h_prev, C_prev)

    assert theta_next.shape == (2,), f"theta_next wrong: {theta_next.shape}"
    assert h_t.shape        == (4,), f"h_t wrong: {h_t.shape}"
    assert C_t.shape        == (4,), f"C_t wrong: {C_t.shape}"
    print(f"Shape check  : PASSED — theta_next={tuple(theta_next.shape)}, "
          f"h_t={tuple(h_t.shape)}, C_t={tuple(C_t.shape)}")

    # 2. Value range
    assert (h_t.abs() < 1.0).all(), f"h_t out of (-1,1): {h_t}"
    print(f"Value range  : PASSED — h_t ∈ (-1,1)")

    # 3. Gradient check: C_prev must be non-zero so W_f gets gradient
    # (f_t * C_prev = 0 when C_prev=zeros → W_f dead in single-step).
    # In actual training, C_prev becomes non-zero after step 1.
    C_prev_nonzero = torch.rand(4)
    theta_next, h_t, C_t = cell(theta_t, y_t, h_prev, C_prev_nonzero)
    (theta_next.sum() + h_t.sum() + C_t.sum()).backward()

    for name, p in gnn.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"No gradient: gnn.{name}"
    for name, p in cell.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"No gradient: cell.{name}"
    print("Gradient check: PASSED — grad flows → GNN + ClassicalLSTMCell")

    # 4. theta_size flexibility check (p=2 → theta_size=4)
    cell4 = ClassicalLSTMCell(theta_size=4, hidden_size=4)
    theta4 = torch.rand(4)
    with torch.no_grad():
        out, _, _ = cell4(theta4, y_t, torch.zeros(4), torch.zeros(4))
    assert out.shape == (4,), f"theta_size=4 output wrong: {out.shape}"
    print(f"Flexibility  : PASSED — theta_size=4 works correctly")

    print("\nClassicalLSTMCell ready.")


if __name__ == "__main__":
    _demo()
