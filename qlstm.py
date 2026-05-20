"""
qlstm.py — Step 4 of the GNN-Conditioned VQC pipeline.

Adapted from rdisipio/qlstm (qlstm_pennylane.py).
Three key changes from the original:
  1. Weights: TorchLayer (fixed nn.Parameter) → plain QNode + ϕ injected from HyperNet.
  2. Input: [h, x_NLP] → [θ_t, y_t, h_{t-1}]; encoding W_in also from HyperNet.
  3. Output: adds θ_{t+1} from VQC₆ alongside the standard (h_t, C_t).

6 VQCs (vs rdisipio's 4): forget, input, update, output, hidden, theta.
Separate device + named wires per VQC — same pattern as rdisipio.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QLSTMCell(nn.Module):
    """
    One QLSTM step: (θ_t, y_t, h_{t-1}, C_{t-1}, hypernet_out) → (θ_{t+1}, h_t, C_t).

    This module has NO learnable parameters — all weights come from HyperNet.

    Args (squeeze batch dim before calling):
        theta_t : (n_qaoa_params,) = (2,)   current QAOA params
        y_t     : scalar or (1,)             QAOA cost ⟨H_C(θ_t)⟩ at step t
        h_prev  : (n_qubits,) = (4,)        previous hidden state
        C_prev  : (n_qubits,) = (4,)        previous cell state
        W_in    : (n_qubits, concat_size)   input projection from HyperNet
        b_in    : (n_qubits,)               input projection bias from HyperNet
        phi_1…phi_6 : (n_qlayers, n_qubits) VQC weights from HyperNet

    Returns:
        theta_next : (2,)  new QAOA parameters (γ, β) ∈ [-1, 1]
        h_t        : (4,)  new hidden state ∈ (-1, 1)
        C_t        : (4,)  new cell state
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_qlayers: int = 2,
        n_qaoa_params: int = 2,
        backend: str = "default.qubit",
        diff_method: str = "backprop",
    ) -> None:
        super().__init__()
        self.n_qubits      = n_qubits
        self.n_qlayers     = n_qlayers
        self.n_qaoa_params = n_qaoa_params
        self._diff_method = diff_method

        gate_names = ["forget", "input", "update", "output", "hidden", "theta"]

        # Separate device + named wires per VQC (same pattern as rdisipio)
        self._wires = {
            name: [f"wire_{name}_{i}" for i in range(n_qubits)]
            for name in gate_names
        }
        self._devs = {
            name: qml.device(backend, wires=self._wires[name])
            for name in gate_names
        }

        def _make_circuit(dev, wires, diff_method):
            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=wires)
                qml.BasicEntanglerLayers(weights, wires=wires)
                return [qml.expval(qml.PauliZ(w)) for w in wires]
            return circuit

        self._circuits = {
            name: _make_circuit(self._devs[name], self._wires[name], diff_method)
            for name in gate_names
        }

    def _vqc(self, name: str, y_enc: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Run named VQC and return (n_qubits,) tensor matching y_enc dtype."""
        out = self._circuits[name](y_enc, phi)
        result = torch.stack(out) if isinstance(out, (list, tuple)) else out
        return result.to(y_enc.dtype)  # PennyLane may return float64; match input

    def forward(
        self,
        theta_t: torch.Tensor,   # (2,)
        y_t:     torch.Tensor,   # scalar or (1,)
        h_prev:  torch.Tensor,   # (4,)
        C_prev:  torch.Tensor,   # (4,)
        W_in:    torch.Tensor,   # (4, 7)
        b_in:    torch.Tensor,   # (4,)
        phi_1:   torch.Tensor,   # (2, 4)
        phi_2:   torch.Tensor,   # (2, 4)
        phi_3:   torch.Tensor,   # (2, 4)
        phi_4:   torch.Tensor,   # (2, 4)
        phi_5:   torch.Tensor,   # (2, 4)
        phi_6:   torch.Tensor,   # (2, 4)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # lightning.gpu (adjoint) requires float64; default.qubit uses W_in dtype
        dtype = torch.float64 if self._diff_method == "adjoint" else W_in.dtype
        theta_t = theta_t.to(dtype)
        y_t     = y_t.to(dtype)
        h_prev  = h_prev.to(dtype)
        C_prev  = C_prev.to(dtype)
        W_in    = W_in.to(dtype)
        b_in    = b_in.to(dtype)
        phi_1   = phi_1.to(dtype)
        phi_2   = phi_2.to(dtype)
        phi_3   = phi_3.to(dtype)
        phi_4   = phi_4.to(dtype)
        phi_5   = phi_5.to(dtype)
        phi_6   = phi_6.to(dtype)

        # Encode: [θ_t(2), y_t(1), h_{t-1}(4)] → (7,) → (n_qubits=4,) angles
        v     = torch.cat([theta_t, y_t.reshape(1), h_prev])  # (7,)
        y_enc = F.linear(v, W_in, b_in)                       # (4,)

        # 4 LSTM gates via VQC 1–4
        f_t = torch.sigmoid(self._vqc("forget", y_enc, phi_1))  # (4,)
        i_t = torch.sigmoid(self._vqc("input",  y_enc, phi_2))  # (4,)
        g_t = torch.tanh   (self._vqc("update", y_enc, phi_3))  # (4,)
        o_t = torch.sigmoid(self._vqc("output", y_enc, phi_4))  # (4,)

        # Classical cell state (same as standard LSTM)
        C_t = f_t * C_prev + i_t * g_t                          # (4,)

        # Hidden state via VQC 5
        h_raw = self._vqc("hidden", y_enc, phi_5)               # (4,)
        h_t   = o_t * torch.tanh(h_raw)                         # (4,) ∈ (-1,1)

        # New QAOA params via VQC 6: take first 2 expvals as (γ, β)
        theta_raw  = self._vqc("theta", y_enc, phi_6)                    # (4,)
        theta_next = theta_raw[:self.n_qaoa_params]               # (n_qaoa_params,)

        return theta_next, h_t, C_t


# ---------------------------------------------------------------------------
# Verification demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Sanity-checks for one QLSTM step:
      1. Output shapes: theta_next=(2,), h_t=(4,), C_t=(4,).
      2. Value range: |h_t| < 1 (tanh-bounded).
      3. Gradient flows theta_next → ϕ₁…ϕ₆ → HyperNet params.
    """
    import networkx as nx
    from gnn import GNNEncoder, nx_to_pyg
    from hypernet import HyperNet

    G = nx.path_graph(6)

    gnn      = GNNEncoder(node_feature_dim=1, dim_h=8, n_qaoa_params=2, qlstm_h_dim=4)
    hypernet = HyperNet(embed_dim=64, hidden_dim=128, n_vqcs=6, n_qlayers=2,
                        n_qubits=4, concat_size=7)
    cell     = QLSTMCell(n_qubits=4, n_qlayers=2)

    print(f"QLSTMCell learnable params: {sum(p.numel() for p in cell.parameters())}")
    print(f"(All weights come from HyperNet — cell has none of its own)\n")

    gnn.train(); hypernet.train()
    theta0, h0, e_G = gnn(nx_to_pyg(G))     # (1,2), (1,4), (1,64)
    out = hypernet(e_G)                      # (W_in, b_in, phi_1,...,phi_6)

    # Squeeze batch dim for QLSTMCell (operates on single graphs)
    theta_t = theta0.squeeze(0)             # (2,)
    h_prev  = h0.squeeze(0)                # (4,)
    C_prev  = torch.zeros(4)
    y_t     = torch.tensor(0.5)            # dummy QAOA cost scalar

    W_in = out[0].squeeze(0)              # (4, 7)
    b_in = out[1].squeeze(0)              # (4,)
    phis = [p.squeeze(0) for p in out[2:]] # 6 × (2, 4)

    # 1. Shape check (no grad)
    with torch.no_grad():
        theta_next, h_t, C_t = cell(theta_t, y_t, h_prev, C_prev, W_in, b_in, *phis)

    assert theta_next.shape == (2,), f"theta_next shape wrong: {theta_next.shape}"
    assert h_t.shape        == (4,), f"h_t shape wrong: {h_t.shape}"
    assert C_t.shape        == (4,), f"C_t shape wrong: {C_t.shape}"
    print(f"Shape check  : PASSED  — theta_next={tuple(theta_next.shape)}, "
          f"h_t={tuple(h_t.shape)}, C_t={tuple(C_t.shape)}")

    # 2. Value range: |h_t| < 1 (o_t ⊙ tanh(h_raw), both bounded)
    assert (h_t.abs() < 1.0).all(), f"h_t out of (-1,1): {h_t}"
    print(f"Value range  : PASSED  — h_t ∈ (-1,1) : {h_t.tolist()}")
    print(f"                         theta_next    : {theta_next.tolist()}")

    # 3. Gradient check: theta_next must reach HyperNet params
    theta_next, h_t, C_t = cell(theta_t, y_t, h_prev, C_prev, W_in, b_in, *phis)
    theta_next.sum().backward()

    for name, p in hypernet.named_parameters():
        assert p.grad is not None,         f"No gradient: hypernet.{name}"
        assert p.grad.abs().sum() > 0,     f"Zero gradient: hypernet.{name}"
    print("Gradient check: PASSED  — grad flows theta_next → phis → HyperNet")

    print("\nQLSTMCell ready.")


if __name__ == "__main__":
    _demo()
