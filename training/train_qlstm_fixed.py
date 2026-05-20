"""
train_qlstm_fixed.py — Train GNN + QLSTMCellFixed (fixed-phi QLSTM baseline).

Architecture: GNN(G) → θ₀, h₀ → QLSTMCellFixed × T
No HyperNet — cell owns its phi weights (same for all graphs).

Purpose: ablation to show that per-instance phi from HyperNet (GNN-QLSTM)
         outperforms fixed phi shared across all graphs (this model).
"""

from __future__ import annotations

import os
import sys

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAINING = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)
os.chdir(_ROOT)

import pickle
import random
import time

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from gnn import GNNEncoder, nx_to_pyg
from qlstm_fixed import QLSTMCellFixed
from qaoa_maxcut import QAOAMaxCut
from train import load_or_create_dataset, maxcut_optimal, CONFIG as TRAIN_CONFIG


CONFIG = {
    **TRAIN_CONFIG,
    'qaoa_p':   2,
    'dim_h':    16,
    'T':        10,
    'ckpt_dir': 'checkpoints',
    'log_every': 5,
}


def run_pipeline_fixed(G, gnn, cell, T, qaoa_p=2, qaoa=None):
    """GNN → QLSTMCellFixed × T. No HyperNet."""
    if qaoa is None:
        qaoa = QAOAMaxCut(G, p=qaoa_p)

    theta0, h0, _ = gnn(nx_to_pyg(G))
    theta_t = theta0.squeeze(0)
    h_t     = h0.squeeze(0)
    C_t     = torch.zeros(cell.n_qubits)

    thetas = []
    for _ in range(T):
        y_t = qaoa.cost_from_theta(theta_t.detach())
        theta_t, h_t, C_t = cell(theta_t, y_t, h_t, C_t)
        thetas.append(theta_t)

    return qaoa, thetas


def evaluate(gnn, cell, val_graphs, T, qaoa_p=2, qaoa_cache=None) -> float:
    gnn.eval(); cell.eval()
    ratios = []
    with torch.no_grad():
        for G in val_graphs:
            opt = maxcut_optimal(G)
            if opt == 0:
                continue
            qaoa_obj = qaoa_cache.get(id(G)) if qaoa_cache else None
            qaoa, thetas = run_pipeline_fixed(G, gnn, cell, T,
                                              qaoa_p=qaoa_p, qaoa=qaoa_obj)
            final_cost = qaoa.cost_from_theta(thetas[-1]).item()
            ratios.append(final_cost / opt)
    return sum(ratios) / len(ratios) if ratios else 0.0


def train(cfg: dict = CONFIG, resume_ckpt: str | None = None) -> None:
    torch.manual_seed(cfg['seed'])
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)

    qaoa_p        = cfg['qaoa_p']
    dim_h         = cfg['dim_h']
    T             = cfg['T']
    n_qaoa_params = qaoa_p * 2
    concat_size   = n_qaoa_params + 1 + 4

    train_graphs, val_graphs = load_or_create_dataset(cfg)

    gnn  = GNNEncoder(node_feature_dim=1, dim_h=dim_h,
                      n_qaoa_params=n_qaoa_params, qlstm_h_dim=4)
    cell = QLSTMCellFixed(n_qubits=4, n_qlayers=2,
                          n_qaoa_params=n_qaoa_params, concat_size=concat_size)

    with torch.no_grad():
        from gnn import nx_to_pyg
        theta0, h0, _ = gnn(nx_to_pyg(train_graphs[0]))
        assert theta0.shape == (1, n_qaoa_params), f"theta0: {theta0.shape}"
        print(f"GNN OK: theta0={tuple(theta0.shape)}, h0={tuple(h0.shape)}")

    n_gnn  = sum(p.numel() for p in gnn.parameters())
    n_cell = sum(p.numel() for p in cell.parameters())
    print(f"Params — GNN: {n_gnn:,}  QLSTMCellFixed: {n_cell:,}")
    print(f"Config — p={qaoa_p}, dim_h={dim_h}, T={T}, "
          f"epochs={cfg['n_epochs']}, lr={cfg['lr']}\n")

    optimizer = optim.Adam(
        list(gnn.parameters()) + list(cell.parameters()),
        lr=cfg['lr'],
    )

    start_epoch    = 1
    best_val_ratio = 0.0
    best_ckpt      = os.path.join(cfg['ckpt_dir'], 'qlstm_fixed_best_p2.pth')
    latest_ckpt    = os.path.join(cfg['ckpt_dir'], 'qlstm_fixed_latest_p2.pth')

    if resume_ckpt and os.path.exists(resume_ckpt):
        ck = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        gnn.load_state_dict(ck['gnn_state_dict'])
        cell.load_state_dict(ck['cell_state_dict'])
        if 'optimizer_state_dict' in ck:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch    = ck['epoch'] + 1
        best_val_ratio = ck.get('val_ratio', 0.0)
        print(f"Resumed from epoch {ck['epoch']}, val={best_val_ratio:.4f}")
        print(f"Continuing from epoch {start_epoch} → {cfg['n_epochs']}\n")

    print("Pre-creating QAOA objects...")
    qaoa_train_cache = {id(G): QAOAMaxCut(G, p=qaoa_p) for G in train_graphs}
    qaoa_val_cache   = {id(G): QAOAMaxCut(G, p=qaoa_p) for G in val_graphs}
    print(f"Done: {len(qaoa_train_cache)} train + {len(qaoa_val_cache)} val\n")

    for epoch in range(start_epoch, cfg['n_epochs'] + 1):
        gnn.train(); cell.train()
        epoch_loss = 0.0
        t0 = time.time()
        random.shuffle(train_graphs)

        for G in train_graphs:
            optimizer.zero_grad()
            qaoa, thetas = run_pipeline_fixed(
                G, gnn, cell, T, qaoa_p=qaoa_p,
                qaoa=qaoa_train_cache[id(G)])

            loss = sum(-qaoa.cost_from_theta(th) for th in thetas)
            loss.backward()
            clip_grad_norm_(
                list(gnn.parameters()) + list(cell.parameters()),
                max_norm=cfg['clip_norm'],
            )
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_graphs)
        elapsed  = time.time() - t0

        torch.save({
            'epoch':                epoch,
            'gnn_state_dict':       gnn.state_dict(),
            'cell_state_dict':      cell.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config':               cfg,
        }, latest_ckpt)

        if epoch % cfg['log_every'] == 0 or epoch == 1:
            val_ratio = evaluate(gnn, cell, val_graphs, T,
                                 qaoa_p=qaoa_p, qaoa_cache=qaoa_val_cache)
            tag = ""
            if val_ratio > best_val_ratio:
                best_val_ratio = val_ratio
                torch.save({
                    'epoch':           epoch,
                    'gnn_state_dict':  gnn.state_dict(),
                    'cell_state_dict': cell.state_dict(),
                    'val_ratio':       val_ratio,
                    'config':          cfg,
                }, best_ckpt)
                tag = "  [saved best]"
            print(f"Epoch {epoch:3d}/{cfg['n_epochs']} | loss={avg_loss:.4f} | "
                  f"val={val_ratio:.4f} | best={best_val_ratio:.4f} | "
                  f"{elapsed:.1f}s{tag}")
        else:
            print(f"Epoch {epoch:3d}/{cfg['n_epochs']} | loss={avg_loss:.4f} | "
                  f"{elapsed:.1f}s")

    print(f"\nTraining done. Best val_ratio={best_val_ratio:.4f}")
    print(f"Saved to {best_ckpt}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str,
                        default='checkpoints/qlstm_fixed_latest_p2.pth')
    parser.add_argument('--scratch', action='store_true')
    args = parser.parse_args()

    resume = None if args.scratch else args.resume
    train(resume_ckpt=resume)
