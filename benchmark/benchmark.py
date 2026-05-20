"""
benchmark.py — Full comparison of GNN-QLSTM against classical optimizers.

Two experiments × two problems × 900 test instances.

Experiment 1 — GNN warm-start:
  All methods receive GNN-predicted θ₀. Isolates contribution of HyperNet ϕ.

Experiment 2 — Random init:
  All methods use random θ₀. QLSTM still receives HyperNet ϕ from GNN.

Problems:
  - MaxCut:  Erdős–Rényi random graphs, P ∈ {2/7,...,6/7}, N ∈ {8..16}
  - SK model: complete graph K_N, J_{ij} ~ N(0, 1/√N)

Comparison unit: iterations (conservative, favorable to gradient baselines).
Note: Adam/SGD etc. require ~5× more circuit evaluations per iteration
      (parameter-shift rule for 2 params), but we compare by iteration count.

R-QAOA is excluded: recursive circuit structure is incompatible with fixed-depth
warm-start initialization (see paper).
"""

from __future__ import annotations

import logging
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pickle
import time
from itertools import product

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from gnn import GNNEncoder, nx_to_pyg
from hypernet import HyperNet
from qlstm import QLSTMCell
from qlstm_fixed import QLSTMCellFixed
from classical_lstm_cell import ClassicalLSTMCell
from qaoa_maxcut import QAOAMaxCut

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not found — convergence plots will be skipped.")


# ---------------------------------------------------------------------------
# Constants (following paper 2505.00561)
# ---------------------------------------------------------------------------

P_VALUES  = [2/7, 3/7, 4/7, 5/7, 6/7]   # edge probabilities for ER graphs
N_VALUES  = list(range(8, 17))             # test node counts {8..16}
N_INSTANCES = 20                           # random instances per (P, N)
T_INFERENCE = 50                           # QAOA iterations per run
BASE_SEED   = 42                           # seed_i = BASE_SEED + instance_idx

QAOA_SIM  = 'lightning.qubit'
QAOA_DIFF = 'adjoint'

GRADIENT_LRS = {
    'Adam':     0.1,
    'SGD':      0.01,
    'RMSProp':  0.01,
    'Adagrad':  0.1,
}

RESULTS_DIR = 'benchmark_results'

# Global logger (set up in main)
log = logging.getLogger('benchmark')


def setup_logger(log_path: str) -> None:
    """Log to both stdout and file simultaneously."""
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Log file: {log_path}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_er_graph(n: int, p: float, seed: int) -> nx.Graph:
    """Erdős–Rényi graph, retry until connected."""
    for attempt in range(1000):
        G = nx.erdos_renyi_graph(n, p, seed=seed + attempt)
        if nx.is_connected(G) and G.number_of_edges() > 0:
            return G
    raise RuntimeError(f"Could not generate connected ER({n},{p}) after 1000 attempts")


def generate_sk_graph(n: int, seed: int) -> nx.Graph:
    """Complete graph K_n with Gaussian weights J_ij ~ N(0, 1/sqrt(n))."""
    rng = np.random.RandomState(seed)
    G = nx.complete_graph(n)
    sigma = 1.0 / np.sqrt(n)
    for u, v in G.edges():
        G[u][v]['weight'] = float(rng.normal(0, sigma))
    return G


# ---------------------------------------------------------------------------
# Optimal value computation
# ---------------------------------------------------------------------------

def maxcut_optimal(G: nx.Graph) -> float:
    """Brute-force MaxCut (supports weighted edges). Feasible for n <= ~20."""
    n = G.number_of_nodes()
    edges = [(u, v, G[u][v].get('weight', 1.0)) for u, v in G.edges()]
    best = float('-inf')
    for mask in range(1 << n):
        cut = sum(w for u, v, w in edges
                  if ((mask >> u) & 1) != ((mask >> v) & 1))
        best = max(best, cut)
    return best


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qlstm(ckpt_path: str = 'checkpoints/model_best_p2.pth'):
    """Load GNN-QLSTM from checkpoint. Reads dims from saved config."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"GNN-QLSTM checkpoint not found: {ckpt_path}")
    ck  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {})
    dim_h         = cfg.get('dim_h', 16)
    qaoa_p        = cfg.get('qaoa_p', 2)
    n_qaoa_params = qaoa_p * 2
    ch_gin        = dim_h * 8
    concat_size   = n_qaoa_params + 1 + 4
    gnn      = GNNEncoder(node_feature_dim=1, dim_h=dim_h,
                          n_qaoa_params=n_qaoa_params, qlstm_h_dim=4)
    hypernet = HyperNet(embed_dim=ch_gin, hidden_dim=128, n_vqcs=6,
                        n_qlayers=2, n_qubits=4, concat_size=concat_size)
    cell     = QLSTMCell(n_qubits=4, n_qlayers=2, n_qaoa_params=n_qaoa_params)
    gnn.load_state_dict(ck['gnn_state_dict'])
    hypernet.load_state_dict(ck['hypernet_state_dict'])
    gnn.eval(); hypernet.eval()
    epoch = ck.get('epoch', '?')
    val   = ck.get('val_ratio', float('nan'))
    print(f"  GNN-QLSTM: epoch={epoch}, val_ratio={val:.4f}, "
          f"p={qaoa_p}, dim_h={dim_h}")
    return gnn, hypernet, cell, qaoa_p, n_qaoa_params


def load_clstm(ckpt_path: str = 'checkpoints/classical_best_p2.pth'):
    """Load GNN-CLSTM from checkpoint. Returns (None, None, 2, 4) if not found."""
    if not os.path.exists(ckpt_path):
        print(f"  GNN-CLSTM checkpoint not found ({ckpt_path}) — skipping.")
        return None, None
    ck  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {})
    dim_h         = cfg.get('dim_h', 16)
    qaoa_p        = cfg.get('qaoa_p', 2)
    n_qaoa_params = qaoa_p * 2
    gnn  = GNNEncoder(node_feature_dim=1, dim_h=dim_h,
                      n_qaoa_params=n_qaoa_params, qlstm_h_dim=4)
    cell = ClassicalLSTMCell(theta_size=n_qaoa_params, hidden_size=4)
    gnn.load_state_dict(ck['gnn_state_dict'])
    cell.load_state_dict(ck['cell_state_dict'])
    gnn.eval(); cell.eval()
    epoch = ck.get('epoch', '?')
    val   = ck.get('val_ratio', float('nan'))
    print(f"  GNN-CLSTM: epoch={epoch}, val_ratio={val:.4f}, "
          f"p={qaoa_p}, dim_h={dim_h}")
    return gnn, cell


def load_qlstm_fixed(ckpt_path: str = 'checkpoints/qlstm_fixed_best_p2.pth'):
    """Load GNN-QLSTM-Fixed from checkpoint. Returns None if not found."""
    if not os.path.exists(ckpt_path):
        print(f"  GNN-QLSTM-Fixed checkpoint not found ({ckpt_path}) — skipping.")
        return None, None
    ck  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {})
    dim_h         = cfg.get('dim_h', 16)
    qaoa_p        = cfg.get('qaoa_p', 2)
    n_qaoa_params = qaoa_p * 2
    concat_size   = n_qaoa_params + 1 + 4
    gnn  = GNNEncoder(node_feature_dim=1, dim_h=dim_h,
                      n_qaoa_params=n_qaoa_params, qlstm_h_dim=4)
    cell = QLSTMCellFixed(n_qubits=4, n_qlayers=2,
                          n_qaoa_params=n_qaoa_params, concat_size=concat_size)
    gnn.load_state_dict(ck['gnn_state_dict'])
    cell.load_state_dict(ck['cell_state_dict'])
    gnn.eval(); cell.eval()
    epoch = ck.get('epoch', '?')
    val   = ck.get('val_ratio', float('nan'))
    print(f"  GNN-QLSTM-Fixed: epoch={epoch}, val_ratio={val:.4f}, "
          f"p={qaoa_p}, dim_h={dim_h}")
    return gnn, cell


# ---------------------------------------------------------------------------
# Optimizer runners  (return trajectory: list[float] of length T)
# ---------------------------------------------------------------------------

def _approx_ratio(qaoa, theta, opt_val):
    cost = qaoa.cost_from_theta(theta.detach()).item()
    return cost / opt_val if opt_val > 0 else 0.0


def run_gradient(G, T, opt_class, lr, theta_init, qaoa_p: int = 2):
    """Gradient-based optimizer: T steps from theta_init. Returns trajectory."""
    qaoa    = QAOAMaxCut(G, p=qaoa_p, sim=QAOA_SIM, diff_method=QAOA_DIFF)
    opt_val = maxcut_optimal(G)
    theta   = theta_init.detach().clone().requires_grad_(True)
    opt     = opt_class([theta], lr=lr)
    traj    = []
    for _ in range(T):
        opt.zero_grad()
        (-qaoa.cost_from_theta(theta)).backward()
        opt.step()
        traj.append(_approx_ratio(qaoa, theta, opt_val))
    return traj


def run_qlstm_gnn_init(G, gnn_q, hypernet, cell, T, qaoa_p: int = 2):
    """GNN-QLSTM, Exp 1: full GNN output (θ₀, h₀, ϕ)."""
    qaoa    = QAOAMaxCut(G, p=qaoa_p, sim=QAOA_SIM, diff_method=QAOA_DIFF)
    opt_val = maxcut_optimal(G)
    with torch.no_grad():
        theta0, h0, e_G = gnn_q(nx_to_pyg(G))
        out    = hypernet(e_G)
        theta_t = theta0.squeeze(0)
        h_t     = h0.squeeze(0)
        C_t     = torch.zeros(4)
        W_in    = out[0].squeeze(0)
        b_in    = out[1].squeeze(0)
        phis    = [p.squeeze(0) for p in out[2:]]
        traj = []
        for _ in range(T):
            y_t = qaoa.cost_from_theta(theta_t)
            theta_t, h_t, C_t = cell(theta_t, y_t, h_t, C_t, W_in, b_in, *phis)
            traj.append(_approx_ratio(qaoa, theta_t, opt_val))
    return traj


def run_qlstm_random_init(G, gnn_q, hypernet, cell, T, seed,
                           qaoa_p: int = 2, n_qaoa_params: int = 4):
    """GNN-QLSTM, Exp 2: random θ₀, h₀=zeros; ϕ still from HyperNet(GNN(G))."""
    qaoa    = QAOAMaxCut(G, p=qaoa_p, sim=QAOA_SIM, diff_method=QAOA_DIFF)
    opt_val = maxcut_optimal(G)
    with torch.no_grad():
        _, _, e_G = gnn_q(nx_to_pyg(G))
        out  = hypernet(e_G)
        torch.manual_seed(seed)
        theta_t = torch.rand(n_qaoa_params) * torch.pi
        h_t     = torch.zeros(4)
        C_t     = torch.zeros(4)
        W_in    = out[0].squeeze(0)
        b_in    = out[1].squeeze(0)
        phis    = [p.squeeze(0) for p in out[2:]]
        traj = []
        for _ in range(T):
            y_t = qaoa.cost_from_theta(theta_t)
            theta_t, h_t, C_t = cell(theta_t, y_t, h_t, C_t, W_in, b_in, *phis)
            traj.append(_approx_ratio(qaoa, theta_t, opt_val))
    return traj


def run_clstm_gnn_init(G, gnn_c, cell_c, T, qaoa_p: int = 2):
    """GNN-CLSTM, Exp 1: GNN θ₀ and h₀, no HyperNet."""
    qaoa    = QAOAMaxCut(G, p=qaoa_p, sim=QAOA_SIM, diff_method=QAOA_DIFF)
    opt_val = maxcut_optimal(G)
    with torch.no_grad():
        theta0, h0, _ = gnn_c(nx_to_pyg(G))
        theta_t = theta0.squeeze(0)
        h_t     = h0.squeeze(0)
        C_t     = torch.zeros(4)
        traj = []
        for _ in range(T):
            y_t = qaoa.cost_from_theta(theta_t)
            theta_t, h_t, C_t = cell_c(theta_t, y_t, h_t, C_t)
            traj.append(_approx_ratio(qaoa, theta_t, opt_val))
    return traj


def run_clstm_random_init(G, cell_c, T, seed,
                           qaoa_p: int = 2, n_qaoa_params: int = 4):
    """GNN-CLSTM, Exp 2: random θ₀, h₀=zeros, no GNN."""
    qaoa    = QAOAMaxCut(G, p=qaoa_p, sim=QAOA_SIM, diff_method=QAOA_DIFF)
    opt_val = maxcut_optimal(G)
    with torch.no_grad():
        torch.manual_seed(seed)
        theta_t = torch.rand(n_qaoa_params) * torch.pi
        h_t     = torch.zeros(4)
        C_t     = torch.zeros(4)
        traj = []
        for _ in range(T):
            y_t = qaoa.cost_from_theta(theta_t)
            theta_t, h_t, C_t = cell_c(theta_t, y_t, h_t, C_t)
            traj.append(_approx_ratio(qaoa, theta_t, opt_val))
    return traj


def run_qlstm_fixed_gnn_init(G, gnn_f, cell_f, T, qaoa_p: int = 2):
    """GNN-QLSTM-Fixed, Exp 1: GNN θ₀ and h₀, fixed phi (no HyperNet)."""
    qaoa    = QAOAMaxCut(G, p=qaoa_p, sim=QAOA_SIM, diff_method=QAOA_DIFF)
    opt_val = maxcut_optimal(G)
    with torch.no_grad():
        theta0, h0, _ = gnn_f(nx_to_pyg(G))
        theta_t = theta0.squeeze(0)
        h_t     = h0.squeeze(0)
        C_t     = torch.zeros(4)
        traj = []
        for _ in range(T):
            y_t = qaoa.cost_from_theta(theta_t)
            theta_t, h_t, C_t = cell_f(theta_t, y_t, h_t, C_t)
            traj.append(_approx_ratio(qaoa, theta_t, opt_val))
    return traj


def run_qlstm_fixed_random_init(G, cell_f, T, seed,
                                 qaoa_p: int = 2, n_qaoa_params: int = 4):
    """GNN-QLSTM-Fixed, Exp 2: random θ₀, h₀=zeros, fixed phi."""
    qaoa    = QAOAMaxCut(G, p=qaoa_p, sim=QAOA_SIM, diff_method=QAOA_DIFF)
    opt_val = maxcut_optimal(G)
    with torch.no_grad():
        torch.manual_seed(seed)
        theta_t = torch.rand(n_qaoa_params) * torch.pi
        h_t     = torch.zeros(4)
        C_t     = torch.zeros(4)
        traj = []
        for _ in range(T):
            y_t = qaoa.cost_from_theta(theta_t)
            theta_t, h_t, C_t = cell_f(theta_t, y_t, h_t, C_t)
            traj.append(_approx_ratio(qaoa, theta_t, opt_val))
    return traj


# ---------------------------------------------------------------------------
# Test dataset generation
# ---------------------------------------------------------------------------

def build_test_dataset(problem: str, cache_dir: str = 'data'):
    """
    Build or load nested dict: data[p_val][n_val] = list of 20 graphs.
    problem: 'maxcut' or 'sk'
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'bench_{problem}_graphs.pkl')

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded {problem} test dataset from {cache_path}")
        return data

    print(f"  Generating {problem} test dataset "
          f"({len(P_VALUES)} P × {len(N_VALUES)} N × {N_INSTANCES} instances)...")
    data = {}
    for p_val in P_VALUES:
        data[p_val] = {}
        for n_val in N_VALUES:
            graphs = []
            for idx in range(N_INSTANCES):
                seed = BASE_SEED + int(p_val * 1000) + n_val * 100 + idx
                if problem == 'maxcut':
                    G = generate_er_graph(n_val, p_val, seed)
                else:
                    G = generate_sk_graph(n_val, seed)
                graphs.append(G)
            data[p_val][n_val] = graphs

    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved to {cache_path}")
    return data


# ---------------------------------------------------------------------------
# Run one full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    exp_name: str,
    problem:  str,
    data:     dict,
    gnn_q, hypernet, qlstm_cell,
    gnn_c,  clstm_cell,       # may be None
    gnn_f=None, fixed_cell=None,  # GNN-QLSTM-Fixed, may be None
    T:             int = T_INFERENCE,
    qaoa_p:        int = 2,
    n_qaoa_params: int = 4,
    ckpt_path:     str | None = None,  # incremental checkpoint path
) -> dict:
    """
    Run all optimizers on all (P, N, instance) combinations.

    Returns:
        results[method][(p_val, n_val)] = np.ndarray of shape (N_INSTANCES, T)
    """
    if exp_name == 'exp1':
        qlstm_name = 'HyperNet_QLSTM_GNN_init'
        fixed_name = 'Fixed_Phi_QLSTM_GNN_init'
        clstm_name = 'CLSTM_GNN_init'
        grad_sfx   = '_GNN_init'
    else:
        qlstm_name = 'HyperNet_QLSTM'
        fixed_name = 'Fixed_Phi_QLSTM'
        clstm_name = 'CLSTM'
        grad_sfx   = ''

    methods = [qlstm_name,
               f'Adam{grad_sfx}', f'SGD{grad_sfx}',
               f'RMSProp{grad_sfx}', f'Adagrad{grad_sfx}']
    if gnn_c is not None:
        methods.insert(1, clstm_name)
    if gnn_f is not None:
        methods.insert(1, fixed_name)

    # Load incremental checkpoint nếu có (resume từ điểm bị crash)
    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            results = pickle.load(f)
        done_keys = set(next(iter(results.values())).keys()) if results else set()
        log.info(f"  Resumed from checkpoint: {len(done_keys)}/{len(P_VALUES)*len(N_VALUES)} done")
    else:
        results   = {m: {} for m in methods}
        done_keys = set()

    total = len(P_VALUES) * len(N_VALUES)
    done  = 0

    for p_val, n_val in product(P_VALUES, N_VALUES):
        graphs = data[p_val][n_val]
        done += 1
        tag = f"[{exp_name.upper()}/{problem.upper()}] [{done:2d}/{total}] P={p_val:.4f} N={n_val}"

        # Skip nếu đã có trong checkpoint
        if (p_val, n_val) in done_keys:
            log.info(f"{tag} SKIPPED (already in checkpoint)")
            continue

        log.info(f"{tag} ...")
        t0 = time.time()

        trajs = {m: [] for m in methods}

        try:
          for idx, G in enumerate(graphs):
            seed_i = BASE_SEED + idx

            # GNN init for gradient-based (Exp 1)
            if exp_name == 'exp1':
                with torch.no_grad():
                    theta0_gnn, _, _ = gnn_q(nx_to_pyg(G))
                    theta_gnn = theta0_gnn.squeeze(0)
            else:
                torch.manual_seed(seed_i)
                theta_gnn = torch.rand(n_qaoa_params) * torch.pi

            # Gradient-based optimizers
            for opt_name, opt_class in [('Adam', optim.Adam),
                                         ('SGD',  optim.SGD),
                                         ('RMSProp', optim.RMSprop),
                                         ('Adagrad', optim.Adagrad)]:
                lr   = GRADIENT_LRS[opt_name]
                traj = run_gradient(G, T, opt_class, lr, theta_gnn, qaoa_p=qaoa_p)
                trajs[f'{opt_name}{grad_sfx}'].append(traj)

            # GNN-QLSTM
            if exp_name == 'exp1':
                traj = run_qlstm_gnn_init(G, gnn_q, hypernet, qlstm_cell, T,
                                           qaoa_p=qaoa_p)
            else:
                traj = run_qlstm_random_init(G, gnn_q, hypernet, qlstm_cell, T,
                                              seed_i, qaoa_p=qaoa_p,
                                              n_qaoa_params=n_qaoa_params)
            trajs[qlstm_name].append(traj)

            # GNN-CLSTM (optional)
            if gnn_c is not None:
                if exp_name == 'exp1':
                    traj = run_clstm_gnn_init(G, gnn_c, clstm_cell, T, qaoa_p=qaoa_p)
                else:
                    traj = run_clstm_random_init(G, clstm_cell, T, seed_i,
                                                  qaoa_p=qaoa_p,
                                                  n_qaoa_params=n_qaoa_params)
                trajs['GNN-CLSTM'].append(traj)

            # GNN-QLSTM-Fixed (optional)
            if gnn_f is not None:
                if exp_name == 'exp1':
                    traj = run_qlstm_fixed_gnn_init(G, gnn_f, fixed_cell, T,
                                                     qaoa_p=qaoa_p)
                else:
                    traj = run_qlstm_fixed_random_init(G, fixed_cell, T, seed_i,
                                                        qaoa_p=qaoa_p,
                                                        n_qaoa_params=n_qaoa_params)
                trajs['GNN-QLSTM-Fixed'].append(traj)

          for m in methods:
              results[m][(p_val, n_val)] = np.array(trajs[m])
          elapsed = time.time() - t0
          log.info(f"{tag} DONE {elapsed:.1f}s")

          # Lưu incremental checkpoint sau mỗi (P,N) — resume nếu crash
          if ckpt_path:
              with open(ckpt_path, 'wb') as f:
                  pickle.dump(results, f)

        except Exception as e:
          elapsed = time.time() - t0
          log.error(f"{tag} ERROR after {elapsed:.1f}s: {e}")
          log.error(traceback.format_exc())

    return results


# ---------------------------------------------------------------------------
# Reporting: summary table at iter k
# ---------------------------------------------------------------------------

def print_table(results: dict, at_iter: int, exp_name: str, problem: str):
    """Print mean ± std table at a given iteration."""
    methods = list(results.keys())
    print(f"\n{'='*70}")
    print(f"  {exp_name.upper()} | {problem.upper()} | Approx ratio at iteration {at_iter}")
    print(f"{'='*70}")

    # Per P-value rows, mean over N
    header = f"{'P':>6}  " + "  ".join(f"{m:>12}" for m in methods)
    print(header)
    print("-" * len(header))

    for p_val in P_VALUES:
        row_vals = []
        for m in methods:
            arrs = []
            for n_val in N_VALUES:
                mat = results[m][(p_val, n_val)]   # (N_INSTANCES, T)
                arrs.append(mat[:, at_iter - 1])    # (N_INSTANCES,)
            all_ratios = np.concatenate(arrs)
            row_vals.append(f"{all_ratios.mean():.3f}±{all_ratios.std():.3f}")
        print(f"  {p_val:.4f}  " + "  ".join(f"{v:>12}" for v in row_vals))

    # Overall mean
    print("-" * len(header))
    overall = []
    for m in methods:
        all_ratios = []
        for p_val in P_VALUES:
            for n_val in N_VALUES:
                all_ratios.append(results[m][(p_val, n_val)][:, at_iter - 1])
        all_ratios = np.concatenate(all_ratios)
        overall.append(f"{all_ratios.mean():.3f}±{all_ratios.std():.3f}")
    print(f"  {'Overall':>6}  " + "  ".join(f"{v:>12}" for v in overall))


def print_per_N_table(results: dict, at_iter: int):
    """Print mean ratio per node size N (averaged over P and instances)."""
    methods = list(results.keys())
    print(f"\n  Per node size N (iter {at_iter}, mean over P and instances):")
    header = f"  {'N':>4} " + " ".join(f"{m:>12}" for m in methods)
    print(header)
    for n_val in N_VALUES:
        row = []
        for m in methods:
            arrs = [results[m][(p_val, n_val)][:, at_iter - 1] for p_val in P_VALUES]
            val = np.concatenate(arrs).mean()
            row.append(f"{val:>12.4f}")
        print(f"  {n_val:>4} " + " ".join(row))


# ---------------------------------------------------------------------------
# Plotting: convergence curves
# ---------------------------------------------------------------------------

def plot_convergence(results: dict, exp_name: str, problem: str,
                     fix_p: float = 3/7):
    """
    9 subplots, one per N ∈ {8..16}. Fix P = fix_p.
    X: iteration (1..T), Y: mean approx ratio, shaded = ± 1 std.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping convergence plots (matplotlib not available).")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    methods = list(results.keys())
    colors  = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=False)
    axes = axes.flatten()
    T = list(results.values())[0][(fix_p, N_VALUES[0])].shape[1]
    x = np.arange(1, T + 1)

    for ax_idx, n_val in enumerate(N_VALUES):
        ax = axes[ax_idx]
        for m_idx, m in enumerate(methods):
            mat  = results[m][(fix_p, n_val)]   # (N_INSTANCES, T)
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)
            c    = colors[m_idx]
            ax.plot(x, mean, label=m, color=c, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=c)
        ax.set_title(f"N={n_val}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if ax_idx >= 6:
            ax.set_xlabel("Iteration", fontsize=9)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Approx ratio", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9,
               bbox_to_anchor=(1.0, 1.0))
    p_str = f"{fix_p:.4f}"
    fig.suptitle(
        f"Convergence curves — {exp_name.upper()} | {problem.upper()} | P={p_str}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    fname = os.path.join(RESULTS_DIR, f"convergence_{exp_name}_{problem}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-clstm', action='store_true',
                        help='Skip GNN-CLSTM')
    parser.add_argument('--no-fixed', action='store_true',
                        help='Skip GNN-QLSTM-Fixed ablation baseline')
    parser.add_argument('--problems', nargs='+', default=['maxcut', 'sk'],
                        choices=['maxcut', 'sk'],
                        help='Problems to run (default: maxcut sk)')
    parser.add_argument('--exps', nargs='+', default=['exp1', 'exp2'],
                        choices=['exp1', 'exp2'],
                        help='Experiments to run (default: exp1 exp2)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logger(os.path.join(RESULTS_DIR, 'benchmark_log.txt'))

    log.info("=" * 60)
    log.info("BENCHMARK: GNN-QLSTM vs Classical Optimizers")
    log.info(f"  T={T_INFERENCE} | N_INSTANCES={N_INSTANCES} | N={N_VALUES} | P={[f'{p:.4f}' for p in P_VALUES]}")
    log.info("=" * 60)

    log.info("Loading models...")
    gnn_q, hypernet, qlstm_cell, qaoa_p, n_qaoa_params = load_qlstm()
    if args.no_clstm:
        log.info("  GNN-CLSTM: skipped (--no-clstm)")
        gnn_c, clstm_cell = None, None
    else:
        gnn_c, clstm_cell = load_clstm()

    if args.no_fixed:
        log.info("  GNN-QLSTM-Fixed: skipped (--no-fixed)")
        gnn_f, fixed_cell = None, None
    else:
        gnn_f, fixed_cell = load_qlstm_fixed()

    for problem in args.problems:
        log.info(f"\n{'─'*60}")
        log.info(f"Problem: {problem.upper()}")

        log.info("Building test dataset...")
        data = build_test_dataset(problem)

        for exp_name in args.exps:
            exp_label = "GNN warm-start" if exp_name == 'exp1' else "Random init"
            log.info(f"\n[{exp_name.upper()}] {exp_label} | {problem.upper()}")
            log.info("Running optimizers...")

            ckpt_path = os.path.join(
                RESULTS_DIR, f'results_{exp_name}_{problem}_ckpt.pkl')
            results = run_experiment(
                exp_name, problem, data,
                gnn_q, hypernet, qlstm_cell,
                gnn_c, clstm_cell,
                gnn_f, fixed_cell,
                qaoa_p=qaoa_p,
                n_qaoa_params=n_qaoa_params,
                ckpt_path=ckpt_path,
            )

            # Save raw results
            cache_path = os.path.join(
                RESULTS_DIR, f'results_{exp_name}_{problem}.pkl'
            )
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
            log.info(f"  Saved: {cache_path}")

            # Tables at iter 3 and iter 50
            for at_iter in [3, T_INFERENCE]:
                print_table(results, at_iter, exp_name, problem)
                print_per_N_table(results, at_iter)

            # Convergence plots (fix P=3/7)
            plot_convergence(results, exp_name, problem, fix_p=3/7)

    log.info(f"\n{'='*60}")
    log.info(f"Benchmark complete. Results saved to: {RESULTS_DIR}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()

