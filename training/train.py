"""
train.py — Step 5: Meta-training loop for GNN-Conditioned VQC pipeline.

Pipeline: GNN(G) → HyperNet → QLSTMCell × T → QAOA
Meta-loss: L = Σ_{t=1..T} -⟨H_C(θ_t)⟩  (BPTT through T steps)
Learnable: GNN + HyperNet params (QLSTMCell has 0 params of its own).
"""

from __future__ import annotations

import os
import sys

# Allow imports from project root (gnn, hypernet, qlstm, qaoa_maxcut)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # keep relative paths (checkpoints/, data/) working

import pickle
import random
import time

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from gnn import GNNEncoder, nx_to_pyg
from hypernet import HyperNet
from qlstm import QLSTMCell
from qaoa_maxcut import QAOAMaxCut


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

P_VALUES = [2/7, 3/7, 4/7, 5/7, 6/7]   # edge probabilities for ER graphs

CONFIG = {
    'n_train':     400,
    'n_val':       100,
    'n_nodes_min': 6,
    'n_nodes_max': 10,
    'T':           10,       # QLSTM steps per graph
    'n_epochs':    100,
    'lr':          1e-3,
    'clip_norm':   1.0,
    'seed':        42,
    'data_dir':    'data',
    'train_data':  'train_graphs_v2.pkl',
    'val_data':    'val_graphs_v2.pkl',
    'ckpt_dir':    'checkpoints',
    'log_every':   5,
    'qaoa_p':      2,        # QAOA circuit depth
    'dim_h':       16,       # GNN hidden dim (ch_gin = dim_h*8)
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def generate_graphs(n_graphs: int, n_min: int, n_max: int, seed: int) -> list:
    """Generate random connected ER graphs with P sampled from P_VALUES."""
    rng = random.Random(seed)
    graphs = []
    attempt = 0
    while len(graphs) < n_graphs:
        n = rng.randint(n_min, n_max)
        p = rng.choice(P_VALUES)           # sample P uniformly from {2/7..6/7}
        G = nx.erdos_renyi_graph(n, p=p, seed=seed + attempt)
        attempt += 1
        if nx.is_connected(G) and G.number_of_edges() > 0:
            graphs.append(G)
    return graphs


def load_or_create_dataset(cfg: dict) -> tuple[list, list]:
    """Load graphs from disk if present, otherwise generate and save."""
    os.makedirs(cfg['data_dir'], exist_ok=True)
    train_path = os.path.join(cfg['data_dir'], cfg.get('train_data', 'train_graphs_v2.pkl'))
    val_path   = os.path.join(cfg['data_dir'], cfg.get('val_data',   'val_graphs_v2.pkl'))

    if os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path, 'rb') as f:
            train_graphs = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_graphs = pickle.load(f)
        print(f"Loaded dataset: {len(train_graphs)} train, {len(val_graphs)} val graphs")
    else:
        print(f"Generating dataset ({cfg['n_train']} train + {cfg['n_val']} val) "
              f"with P ∈ {P_VALUES} ...")
        train_graphs = generate_graphs(cfg['n_train'], cfg['n_nodes_min'],
                                       cfg['n_nodes_max'], seed=cfg['seed'])
        val_graphs   = generate_graphs(cfg['n_val'],   cfg['n_nodes_min'],
                                       cfg['n_nodes_max'], seed=cfg['seed'] + 10000)
        with open(train_path, 'wb') as f:
            pickle.dump(train_graphs, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_graphs, f)
        print(f"Saved to {train_path}, {val_path}")

    return train_graphs, val_graphs


# ---------------------------------------------------------------------------
# MaxCut brute-force (exact, feasible for n <= ~20 nodes)
# ---------------------------------------------------------------------------

def maxcut_optimal(G: nx.Graph) -> int:
    """Exact MaxCut value via brute-force over all 2^n bipartitions."""
    n = G.number_of_nodes()
    edges = list(G.edges())
    best = 0
    for mask in range(1 << n):
        cut = sum(1 for u, v in edges if ((mask >> u) & 1) != ((mask >> v) & 1))
        if cut > best:
            best = cut
    return best


# ---------------------------------------------------------------------------
# One forward pass through the full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(G, gnn, hypernet, cell, T, train_mode=True, qaoa_p=1,
                 qaoa_backend="default.qubit", qaoa_diff_method="backprop",
                 qaoa=None):
    """
    Run GNN → HyperNet → QLSTM × T for graph G.

    Pass a pre-created qaoa object to avoid rebuilding the QNode every call.
    """
    if qaoa is None:
        qaoa = QAOAMaxCut(G, p=qaoa_p, sim=qaoa_backend, diff_method=qaoa_diff_method)

    theta0, h0, e_G = gnn(nx_to_pyg(G))
    out = hypernet(e_G)

    theta_t = theta0.squeeze(0)
    h_t     = h0.squeeze(0)
    C_t     = torch.zeros(4)
    W_in    = out[0].squeeze(0)
    b_in    = out[1].squeeze(0)
    phis    = [p.squeeze(0) for p in out[2:]]   # 6 × (2, 4)

    thetas = []
    for _ in range(T):
        # y_t: QAOA cost as feedback signal (detach → stop gradient through y_t)
        y_t = qaoa.cost_from_theta(theta_t.detach())
        theta_t, h_t, C_t = cell(theta_t, y_t, h_t, C_t, W_in, b_in, *phis)
        thetas.append(theta_t)

    return qaoa, thetas


# ---------------------------------------------------------------------------
# Validation: mean approximation ratio over val_graphs
# ---------------------------------------------------------------------------

def evaluate(gnn, hypernet, cell, val_graphs, T, qaoa_p=1,
             qaoa_backend="default.qubit", qaoa_diff_method="backprop",
             qaoa_cache=None) -> float:
    gnn.eval(); hypernet.eval()
    ratios = []
    with torch.no_grad():
        for G in val_graphs:
            opt = maxcut_optimal(G)
            if opt == 0:
                continue
            qaoa_obj = qaoa_cache.get(id(G)) if qaoa_cache else None
            qaoa, thetas = run_pipeline(G, gnn, hypernet, cell, T,
                                        train_mode=False, qaoa_p=qaoa_p,
                                        qaoa_backend=qaoa_backend,
                                        qaoa_diff_method=qaoa_diff_method,
                                        qaoa=qaoa_obj)
            final_cost = qaoa.cost_from_theta(thetas[-1]).item()
            ratios.append(final_cost / opt)
    return sum(ratios) / len(ratios) if ratios else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict = CONFIG, resume_ckpt: str | None = None) -> None:
    torch.manual_seed(cfg['seed'])
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)

    # Dataset
    train_graphs, val_graphs = load_or_create_dataset(cfg)

    # Derive model dims from config
    qaoa_p        = cfg['qaoa_p']
    dim_h         = cfg['dim_h']
    n_qaoa_params = qaoa_p * 2
    ch_gin        = dim_h * 8
    concat_size   = n_qaoa_params + 1 + 4   # theta + y + h

    # Models
    gnn      = GNNEncoder(node_feature_dim=1, dim_h=dim_h,
                          n_qaoa_params=n_qaoa_params, qlstm_h_dim=4)
    hypernet = HyperNet(embed_dim=ch_gin, hidden_dim=128, n_vqcs=6,
                        n_qlayers=2, n_qubits=4, concat_size=concat_size)
    cell     = QLSTMCell(n_qubits=4, n_qlayers=2, n_qaoa_params=n_qaoa_params)

    n_gnn   = sum(p.numel() for p in gnn.parameters())
    n_hyper = sum(p.numel() for p in hypernet.parameters())
    n_cell  = sum(p.numel() for p in cell.parameters())
    print(f"\nModel params — GNN: {n_gnn:,}  HyperNet: {n_hyper:,}  Cell: {n_cell}")
    print(f"Training: T={cfg['T']} steps, {cfg['n_epochs']} epochs, lr={cfg['lr']}\n")

    # Optimizer — only GNN + HyperNet (cell has no params)
    optimizer = optim.Adam(
        list(gnn.parameters()) + list(hypernet.parameters()),
        lr=cfg['lr'],
    )

    # Resume from checkpoint if provided
    start_epoch    = 1
    best_val_ratio = 0.0
    if resume_ckpt and os.path.exists(resume_ckpt):
        ck = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        gnn.load_state_dict(ck['gnn_state_dict'])
        hypernet.load_state_dict(ck['hypernet_state_dict'])
        if 'optimizer_state_dict' in ck:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch    = ck['epoch'] + 1
        best_val_ratio = ck.get('val_ratio', 0.0)
        print(f"Resumed from checkpoint: epoch={ck['epoch']}, "
              f"val_ratio={best_val_ratio:.4f}")
        print(f"Continuing from epoch {start_epoch} → {cfg['n_epochs']}\n")
    elif resume_ckpt:
        print(f"Warning: checkpoint not found at {resume_ckpt}, training from scratch.\n")

    T = cfg['T']

    for epoch in range(start_epoch, cfg['n_epochs'] + 1):
        gnn.train(); hypernet.train()
        epoch_loss = 0.0
        t0 = time.time()

        random.shuffle(train_graphs)

        for G in train_graphs:
            optimizer.zero_grad()

            qaoa, thetas = run_pipeline(G, gnn, hypernet, cell, T)

            # Meta-loss: sum of -<H_C(θ_t)> over all T steps
            loss = torch.zeros(1)
            for theta_t in thetas:
                loss = loss + (-qaoa.cost_from_theta(theta_t))

            loss.backward()
            clip_grad_norm_(
                list(gnn.parameters()) + list(hypernet.parameters()),
                max_norm=cfg['clip_norm'],
            )
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_graphs)
        elapsed  = time.time() - t0

        # Validation every log_every epochs
        if epoch % cfg['log_every'] == 0 or epoch == 1:
            val_ratio = evaluate(gnn, hypernet, cell, val_graphs, T)
            saved_tag = ""
            if val_ratio > best_val_ratio:
                best_val_ratio = val_ratio
                torch.save({
                    'epoch':               epoch,
                    'gnn_state_dict':      gnn.state_dict(),
                    'hypernet_state_dict': hypernet.state_dict(),
                    'val_ratio':           val_ratio,
                    'config':              cfg,
                }, os.path.join(cfg['ckpt_dir'], 'model_best.pth'))
                saved_tag = "  [saved best]"
            print(f"Epoch {epoch:3d}/{cfg['n_epochs']} | "
                  f"loss={avg_loss:8.4f} | "
                  f"val_ratio={val_ratio:.4f} | "
                  f"best={best_val_ratio:.4f} | "
                  f"{elapsed:.1f}s{saved_tag}")
        else:
            print(f"Epoch {epoch:3d}/{cfg['n_epochs']} | "
                  f"loss={avg_loss:8.4f} | "
                  f"{elapsed:.1f}s")

        # Always save latest
        torch.save({
            'epoch':               epoch,
            'gnn_state_dict':      gnn.state_dict(),
            'hypernet_state_dict': hypernet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config':              cfg,
        }, os.path.join(cfg['ckpt_dir'], 'model_latest.pth'))

    print(f"\nTraining done. Best val_ratio={best_val_ratio:.4f}")
    print(f"Model saved to {cfg['ckpt_dir']}/model_best.pth")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='checkpoints/model_latest.pth',
                        help='Checkpoint path to resume from (default: model_latest.pth)')
    parser.add_argument('--scratch', action='store_true',
                        help='Train from scratch (ignore --resume)')
    args = parser.parse_args()

    resume = None if args.scratch else args.resume
    train(resume_ckpt=resume)
