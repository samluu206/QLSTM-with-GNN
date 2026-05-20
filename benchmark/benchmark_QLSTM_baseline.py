"""
benchmark_QLSTM_baseline.py — Evaluate GNN-QLSTM-Fixed ablation baseline.

Runs only GNN-QLSTM-Fixed (fixed phi, no HyperNet) across all 4 experiments.
Results saved to benchmark_results/results_{exp}_{problem}_fixed.pkl
for later merging with full benchmark results via merge_results.py.

Purpose: prove that per-instance phi from HyperNet (GNN-QLSTM) outperforms
         fixed phi shared across all graphs (GNN-QLSTM-Fixed).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import pickle
import time
from itertools import product

import numpy as np

# Load benchmark.py from the same directory
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location("bm", os.path.join(_DIR, "benchmark.py"))
bm    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

os.chdir(_ROOT)


def run_fixed_experiment(
    exp_name: str,
    problem:  str,
    data:     dict,
    gnn_f,
    fixed_cell,
    T:             int = bm.T_INFERENCE,
    qaoa_p:        int = 2,
    n_qaoa_params: int = 4,
) -> dict:
    """Run GNN-QLSTM-Fixed only on all (P, N, instance) combinations."""
    init_tag   = 'GNN_init' if exp_name == 'exp1' else 'random_init'
    fixed_name = f'HyperNet_QLSTM_fixed_phi_{init_tag}'
    results    = {fixed_name: {}}
    total   = len(bm.P_VALUES) * len(bm.N_VALUES)
    done    = 0

    for p_val, n_val in product(bm.P_VALUES, bm.N_VALUES):
        graphs = data[p_val][n_val]
        done  += 1
        tag    = (f"[{exp_name.upper()}/{problem.upper()}]"
                  f" [{done:2d}/{total}] P={p_val:.4f} N={n_val}")
        bm.log.info(f"{tag} ...")
        t0 = time.time()

        trajs = []
        try:
            for idx, G in enumerate(graphs):
                seed_i = bm.BASE_SEED + idx
                if exp_name == 'exp1':
                    traj = bm.run_qlstm_fixed_gnn_init(
                        G, gnn_f, fixed_cell, T, qaoa_p=qaoa_p)
                else:
                    traj = bm.run_qlstm_fixed_random_init(
                        G, fixed_cell, T, seed_i,
                        qaoa_p=qaoa_p, n_qaoa_params=n_qaoa_params)
                trajs.append(traj)

            results[fixed_name][(p_val, n_val)] = np.array(trajs)
            elapsed = time.time() - t0
            bm.log.info(f"{tag} DONE {elapsed:.1f}s")

        except Exception as e:
            import traceback
            elapsed = time.time() - t0
            bm.log.error(f"{tag} ERROR after {elapsed:.1f}s: {e}")
            bm.log.error(traceback.format_exc())

    return results


def main():
    os.makedirs(bm.RESULTS_DIR, exist_ok=True)
    bm.setup_logger(os.path.join(bm.RESULTS_DIR, 'benchmark_fixed_log.txt'))

    bm.log.info("=" * 60)
    bm.log.info("BENCHMARK ABLATION: GNN-QLSTM-Fixed (fixed phi, no HyperNet)")
    bm.log.info(f"  T={bm.T_INFERENCE} | N_INSTANCES={bm.N_INSTANCES}")
    bm.log.info("=" * 60)

    bm.log.info("Loading GNN-QLSTM-Fixed...")
    gnn_f, fixed_cell = bm.load_qlstm_fixed()
    if gnn_f is None:
        bm.log.error("Checkpoint not found. Train first: python benchmark/train_qlstm_fixed.py")
        return

    cfg           = fixed_cell.__dict__  # read dims from cell
    qaoa_p        = fixed_cell.n_qaoa_params // 2
    n_qaoa_params = fixed_cell.n_qaoa_params

    for problem in ['maxcut', 'sk']:
        bm.log.info(f"\n{'─'*60}")
        bm.log.info(f"Problem: {problem.upper()}")

        bm.log.info("Building test dataset...")
        data = bm.build_test_dataset(problem)

        for exp_name in ['exp1', 'exp2']:
            exp_label = "GNN warm-start" if exp_name == 'exp1' else "Random init"
            bm.log.info(f"\n[{exp_name.upper()}] {exp_label} | {problem.upper()}")

            results = run_fixed_experiment(
                exp_name, problem, data,
                gnn_f, fixed_cell,
                qaoa_p=qaoa_p, n_qaoa_params=n_qaoa_params,
            )

            out_path = os.path.join(
                bm.RESULTS_DIR, f'results_{exp_name}_{problem}_fixed.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump(results, f)
            bm.log.info(f"  Saved: {out_path}")

            bm.print_table(results, at_iter=3,              exp_name=exp_name, problem=problem)
            bm.print_table(results, at_iter=bm.T_INFERENCE, exp_name=exp_name, problem=problem)
            bm.print_per_N_table(results, at_iter=bm.T_INFERENCE)

    bm.log.info(f"\n{'='*60}")
    bm.log.info("Done. Run merge_results.py to combine with full benchmark.")
    bm.log.info("=" * 60)


if __name__ == '__main__':
    main()
