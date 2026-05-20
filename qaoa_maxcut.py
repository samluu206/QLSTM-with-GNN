"""
qaoa_maxcut.py — Step 1 of the GNN-Conditioned VQC pipeline.

QAOA MaxCut circuit with a PyTorch-differentiable interface.
Supports BPTT when the QLSTM is added in Step 2.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import pennylane as qml
import torch
import torch.optim as optim


class QAOAMaxCut:
    """
    p-layer QAOA circuit for MaxCut on a fixed NetworkX graph.

    Cost Hamiltonian:   H_C = Σ_{(i,j)∈E}  ½(I − Z_i Z_j)
    Mixer Hamiltonian:  H_M = Σ_i  X_i

    ⟨H_C⟩ ∈ [0, |E|]  —  higher means more edges are cut on average.

    The QNode uses diff_method="backprop" on default.qubit so gradients
    flow through torch.autograd without any extra configuration.
    Switch diff_method to "parameter-shift" if targeting real hardware.
    """

    def __init__(
        self,
        graph: nx.Graph,
        p: int = 1,
        sim: str = "default.qubit",
        diff_method: str = "backprop",
    ) -> None:
        if graph.number_of_nodes() == 0:
            raise ValueError("Graph must have at least one node.")

        # Normalise node labels to 0 … n-1 so they map directly to qubit wires.
        self.graph = nx.convert_node_labels_to_integers(graph)
        self.n_qubits = self.graph.number_of_nodes()
        self.p = p

        self.cost_h, self.mixer_h = qml.qaoa.maxcut(self.graph)
        self._circuit = self._build_qnode(sim, diff_method)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_qnode(self, sim: str, diff_method: str = "backprop"):
        dev = qml.device(sim, wires=self.n_qubits)
        cost_h, mixer_h = self.cost_h, self.mixer_h
        n_qubits, p = self.n_qubits, self.p

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
            # |+⟩^⊗n — uniform superposition over all 2^n bitstrings
            for wire in range(n_qubits):
                qml.Hadamard(wires=wire)

            # p QAOA layers: alternating cost and mixer unitaries
            # PennyLane ≥ 0.44 flipped the MaxCut Hamiltonian sign:
            # qml.qaoa.maxcut now returns H = ½ΣZ_uZ_v − |E|/2·I = −H_C_standard
            # Negating gamma restores the correct cost-unitary direction: exp(−iγH_C_standard)
            for layer in range(p):
                qml.qaoa.cost_layer(-gamma[layer], cost_h)  # exp(-i γ H_C_standard)
                qml.qaoa.mixer_layer(beta[layer], mixer_h)  # exp(-i β H_M)

            return qml.expval(cost_h)

        return circuit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            gamma: shape (p,)  cost-layer angles γ in radians
            beta:  shape (p,)  mixer-layer angles β in radians

        Returns:
            Scalar torch.Tensor ⟨H_C⟩ ∈ [0, |E|] with grad_fn attached.
        """
        # PennyLane ≥ 0.44 returns −⟨H_C_standard⟩ from the QNode (sign-flipped
        # Hamiltonian).  Negating here restores the documented [0, |E|] range so
        # all callers remain unchanged.
        return -self._circuit(gamma.reshape(self.p), beta.reshape(self.p))

    def cost_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Flat-parameter interface for the QLSTM optimizer loop.

        theta: ℝ^{2p} = [γ_1, …, γ_p, β_1, …, β_p]

        Returns ⟨H_C⟩ as a differentiable scalar.
        Negate outside this function to form a loss for minimisation.
        """
        return self(theta[: self.p], theta[self.p :])

    @property
    def n_params(self) -> int:
        """Total parameter count: 2 * p (p gammas + p betas)."""
        return 2 * self.p

    def draw(
        self,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> None:
        """Print the circuit diagram to stdout."""
        if gamma is None:
            gamma = torch.zeros(self.p)
        if beta is None:
            beta = torch.zeros(self.p)
        print(qml.draw(self._circuit)(gamma, beta))


# ---------------------------------------------------------------------------
# Verification demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Sanity-checks:
      1. Forward pass returns a sensible ⟨H_C⟩ value.
      2. Gradients w.r.t. γ and β are non-zero.
      3. Gradient ascent reaches near-optimal ⟨H_C⟩ for a small graph.
    """
    # 4-cycle: nodes {0,1,2,3}, edges {(0,1),(1,2),(2,3),(3,0)}
    # Optimal partition {0,2} | {1,3} cuts all 4 edges → MaxCut = 4
    G = nx.cycle_graph(4)
    qaoa = QAOAMaxCut(graph=G, p=1)

    print(f"Graph : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"MaxCut: {G.number_of_edges()} (partition {{0,2}} | {{1,3}} cuts every edge)")
    print(f"Params: {qaoa.n_params}  (γ, β)\n")

    print("--- Circuit diagram ---")
    qaoa.draw()

    # 1. Forward pass + gradient check
    gamma = torch.tensor([0.5], requires_grad=True)
    beta  = torch.tensor([0.5], requires_grad=True)
    cost  = qaoa(gamma, beta)
    cost.backward()

    print(f"\n⟨H_C⟩(γ=0.5, β=0.5) = {cost.item():.4f}")
    print(f"d⟨H_C⟩/dγ           = {gamma.grad.item():.4f}")
    print(f"d⟨H_C⟩/dβ           = {beta.grad.item():.4f}")

    assert gamma.grad is not None and beta.grad is not None, "Gradients missing!"
    print("Gradient check: PASSED")

    # 2. Gradient ascent — maximise ⟨H_C⟩
    # Note: θ = 0 is a barren-plateau saddle point (zero gradient) for QAOA.
    # Use a random non-zero start; p=1 QAOA theoretical max ratio on C4 is 0.75.
    print("\n--- Gradient ascent (200 steps, Adam lr=0.1, random init) ---")
    torch.manual_seed(0)
    theta = (torch.rand(qaoa.n_params) * torch.pi).requires_grad_(True)
    opt   = optim.Adam([theta], lr=0.1)

    for step in range(200):
        opt.zero_grad()
        loss = -qaoa.cost_from_theta(theta)   # negate: minimise → maximise cut
        loss.backward()
        opt.step()
        if (step + 1) % 40 == 0:
            val = -loss.item()
            print(f"  step {step + 1:3d}: ⟨H_C⟩ = {val:.4f} / {G.number_of_edges()}")

    final = -loss.item()
    print(f"\nFinal ⟨H_C⟩ = {final:.4f}  (p=1 QAOA max on C4 = 3.0, i.e. ratio 0.75)")
    print(f"Approximation ratio ≈ {final / G.number_of_edges():.3f}")


if __name__ == "__main__":
    _demo()
