"""
merge_results.py — Merge full benchmark results with GNN-QLSTM-Fixed ablation.

Loads:
  benchmark_results/results_{exp}_{problem}.pkl       ← from Colab (all methods)
  benchmark_results/results_{exp}_{problem}_fixed.pkl ← from benchmark_QLSTM_baseline.py

Merges dicts, prints full comparison table, saves plots.

Usage:
  python benchmark/merge_results.py
  python benchmark/merge_results.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys

# Load benchmark.py
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location("bm", os.path.join(_DIR, "benchmark.py"))
bm    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

os.chdir(_ROOT)


RENAME_EXP1 = {
    'GNN-QLSTM':  'HyperNet_QLSTM_GNN_init',
    'GNN-CLSTM':  'CLSTM_GNN_init',
    'Adam':       'Adam_GNN_init',
    'SGD':        'SGD_GNN_init',
    'RMSProp':    'RMSProp_GNN_init',
    'Adagrad':    'Adagrad_GNN_init',
    # Fixed baseline key from benchmark_QLSTM_baseline.py
    'HyperNet_QLSTM_fixed_phi_GNN_init': 'Fixed_Phi_QLSTM_GNN_init',
}
RENAME_EXP2 = {
    'GNN-QLSTM':  'HyperNet_QLSTM',
    'GNN-CLSTM':  'CLSTM',
    # Adam/SGD/RMSProp/Adagrad giữ nguyên trong Exp2
    # Fixed baseline key from benchmark_QLSTM_baseline.py
    'HyperNet_QLSTM_fixed_phi_random_init': 'Fixed_Phi_QLSTM',
}

def rename_methods(results: dict, exp_name: str) -> dict:
    """Rename pkl keys sang naming convention đã quy ước."""
    mapping = RENAME_EXP1 if exp_name == 'exp1' else RENAME_EXP2
    return {mapping.get(k, k): v for k, v in results.items()}


def load_pkl(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_convergence_exp1_vs_exp2(
    exp1_results: dict,
    exp2_results: dict,
    problem: str,
    rdir: str,
    fix_p: float = 3/7,
) -> None:
    """
    So sánh tốc độ hội tụ Exp1 (GNN_init) vs Exp2 (random_init) cho cùng method.

    Layout: 9 subplots (N=8..16), fix P=fix_p.
    Mỗi method: solid = GNN_init (Exp1), dashed = random_init (Exp2).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        bm.log.warning("matplotlib not available — skipping convergence_exp1_vs_exp2 plot")
        return

    # Map Exp1 key → base name (strip _GNN_init suffix)
    def base_name(key: str) -> str:
        return key.replace('_GNN_init', '')

    # Build mapping: base_name → (exp1_data, exp2_data)
    exp1_by_base = {base_name(k): v for k, v in exp1_results.items()}
    exp2_by_base = {k: v for k, v in exp2_results.items()}
    common = [m for m in exp1_by_base if m in exp2_by_base]

    if not common:
        bm.log.warning("No common methods between Exp1 and Exp2 — skipping cross-exp plot")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(common)))
    N_VALUES = bm.N_VALUES
    T = bm.T_INFERENCE
    x = np.arange(1, T + 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    for ax_idx, n_val in enumerate(N_VALUES):
        ax = axes[ax_idx]
        for m_idx, method in enumerate(common):
            c = colors[m_idx]

            # Exp1 (GNN_init) — solid
            arrs1 = [exp1_by_base[method][(fix_p, n_val)]
                     for p in [fix_p] if (fix_p, n_val) in exp1_by_base[method]]
            if arrs1:
                mean1 = arrs1[0].mean(axis=0)
                ax.plot(x, mean1, color=c, linewidth=1.8, linestyle='-',
                        label=f'{method} (GNN_init)')

            # Exp2 (random_init) — dashed
            arrs2 = [exp2_by_base[method][(fix_p, n_val)]
                     for p in [fix_p] if (fix_p, n_val) in exp2_by_base[method]]
            if arrs2:
                mean2 = arrs2[0].mean(axis=0)
                ax.plot(x, mean2, color=c, linewidth=1.8, linestyle='--',
                        label=f'{method} (random_init)')

        ax.set_title(f"N={n_val}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if ax_idx >= 6:
            ax.set_xlabel("Step", fontsize=9)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Approx ratio", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8,
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(
        f"Convergence: GNN_init (solid) vs random_init (dashed) | {problem.upper()} | P={fix_p:.4f}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    fname = os.path.join(rdir, f'convergence_exp1_vs_exp2_{problem}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    bm.log.info(f"  Saved: {fname}")


def _plot_combined_exp(exp_name: str, rdir: str, fix_p: float = 3/7) -> None:
    """
    1 combined plot per experiment: 2 rows (MaxCut, SK) × 9 cols (N=8..16).
    Saved as convergence_{exp_name}_combined.png
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    problems = ['maxcut', 'sk']
    data = {}
    for problem in problems:
        path = os.path.join(rdir, f'results_{exp_name}_{problem}_merged.pkl')
        d = load_pkl(path)
        if d is None:
            bm.log.warning(f"Missing {path} — skipping combined plot for {exp_name}")
            return
        data[problem] = d

    methods  = list(data['maxcut'].keys())
    colors   = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    N_VALUES = bm.N_VALUES
    T        = bm.T_INFERENCE
    x        = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 9, figsize=(28, 8), sharex=True)

    for row, problem in enumerate(problems):
        results = data[problem]
        for col, n_val in enumerate(N_VALUES):
            ax = axes[row, col]
            for m_idx, method in enumerate(methods):
                if (fix_p, n_val) not in results[method]:
                    continue
                mat  = results[method][(fix_p, n_val)]
                mean = mat.mean(axis=0)
                std  = mat.std(axis=0)
                c    = colors[m_idx]
                ax.plot(x, mean, color=c, linewidth=1.5, label=method)
                ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=c)
            ax.set_ylim(0, 1.05) if problem == 'maxcut' else None
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"N={n_val}", fontsize=9)
            if col == 0:
                ax.set_ylabel(problem.upper(), fontsize=10, fontweight='bold')
            if row == 1:
                ax.set_xlabel("Step", fontsize=8)

    init_label = "GNN warm-start" if exp_name == 'exp1' else "Random init"
    fig.suptitle(f"Convergence — {exp_name.upper()} ({init_label}) | P={fix_p:.4f}",
                 fontsize=13, y=1.01)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8,
               bbox_to_anchor=(1.0, 1.0), ncol=1)
    plt.tight_layout()

    fname = os.path.join(rdir, f'convergence_{exp_name}_combined.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    bm.log.info(f"  Saved: {fname}")


def save_summary_table(rdir: str, out_dir: str) -> None:
    """Ghi bảng overall ratio tại iter 3 và iter 50 ra result/summary_ratios.txt."""
    import numpy as np

    exp_labels = {'exp1': 'GNN warm-start', 'exp2': 'Random init'}
    lines = ['=' * 70, 'BENCHMARK SUMMARY — Approximation Ratio', '=' * 70, '']

    for problem in ['maxcut', 'sk']:
        for exp_name in ['exp1', 'exp2']:
            path = os.path.join(rdir, f'results_{exp_name}_{problem}_merged.pkl')
            results = load_pkl(path)
            if results is None:
                continue

            label = exp_labels[exp_name]
            unit  = 'ratio' if problem == 'maxcut' else 'cut value'
            lines.append(f'=== {problem.upper()} | {exp_name.upper()} ({label}) ===')

            for at_iter in [3, bm.T_INFERENCE]:
                lines.append(f'  At iteration {at_iter}:')
                max_len = max(len(m) for m in results)
                for method, data in results.items():
                    all_vals = np.concatenate([
                        arr[:, at_iter - 1] for arr in data.values()
                    ])
                    mean, std = all_vals.mean(), all_vals.std()
                    lines.append(f'    {method:{max_len}s} : {mean:.4f} ± {std:.4f}')
                lines.append('')
            lines.append('')

    out_path = os.path.join(out_dir, 'summary_ratios.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    bm.log.info(f"  Saved: {out_path}")


def _plot_single(problem: str, exp_name: str, rdir: str, out_dir: str,
                  fix_p: float = 3/7) -> None:
    """1 row × 9 cols (N=8..16) for one problem × one experiment."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    path = os.path.join(rdir, f'results_{exp_name}_{problem}_merged.pkl')
    results = load_pkl(path)
    if results is None:
        bm.log.warning(f"Missing {path} — skipping")
        return

    methods  = list(results.keys())
    colors   = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    N_VALUES = bm.N_VALUES
    T        = bm.T_INFERENCE
    x        = np.arange(1, T + 1)

    fig, axes = plt.subplots(1, 9, figsize=(28, 4), sharex=True)

    for col, n_val in enumerate(N_VALUES):
        ax = axes[col]
        for m_idx, method in enumerate(methods):
            if (fix_p, n_val) not in results[method]:
                continue
            mat  = results[method][(fix_p, n_val)]
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)
            c    = colors[m_idx]
            ax.plot(x, mean, color=c, linewidth=1.5, label=method)
            ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=c)
        if problem == 'maxcut':
            ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"N={n_val}", fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        if col == 0:
            ax.set_ylabel("Approx ratio" if problem == 'maxcut' else "Cut value",
                          fontsize=9)

    init_label = "GNN warm-start" if exp_name == 'exp1' else "Random init"
    fig.suptitle(f"Convergence — {problem.upper()} | {exp_name.upper()} ({init_label}) | P={fix_p:.4f}",
                 fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8,
               bbox_to_anchor=(1.0, 1.0), ncol=1)
    plt.tight_layout()

    fname = os.path.join(out_dir, f'convergence_{problem}_{exp_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    bm.log.info(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default=bm.RESULTS_DIR)
    args = parser.parse_args()

    rdir = args.results_dir
    bm.setup_logger(os.path.join(rdir, 'merge_log.txt'))
    bm.log.info("=" * 60)
    bm.log.info("MERGE: Full benchmark + GNN-QLSTM-Fixed ablation")
    bm.log.info("=" * 60)

    for problem in ['maxcut', 'sk']:
        for exp_name in ['exp1', 'exp2']:
            base_path  = os.path.join(rdir, f'results_{exp_name}_{problem}.pkl')
            fixed_path = os.path.join(rdir, f'results_{exp_name}_{problem}_fixed.pkl')

            base  = load_pkl(base_path)
            fixed = load_pkl(fixed_path)

            if base is None:
                bm.log.warning(f"Missing: {base_path} — skipping")
                continue
            if fixed is None:
                bm.log.warning(f"Missing: {fixed_path} — run benchmark_QLSTM_baseline.py first")
                continue

            # Rename theo naming convention trước khi merge
            base  = rename_methods(base,  exp_name)
            fixed = rename_methods(fixed, exp_name)

            # Merge: thứ tự hiển thị — HyperNet_QLSTM trước, Fixed ngay sau
            merged = {**base, **fixed}
            qlstm_key = 'HyperNet_QLSTM_GNN_init' if exp_name == 'exp1' else 'HyperNet_QLSTM'
            fixed_key = 'Fixed_Phi_QLSTM_GNN_init' if exp_name == 'exp1' else 'Fixed_Phi_QLSTM'
            ordered_keys = list(base.keys())
            if fixed_key in fixed and fixed_key not in ordered_keys:
                idx = ordered_keys.index(qlstm_key) + 1 if qlstm_key in ordered_keys else 0
                ordered_keys.insert(idx, fixed_key)
            merged = {k: merged[k] for k in ordered_keys if k in merged}

            bm.log.info(f"\n[{exp_name.upper()}] {problem.upper()} — methods: {list(merged.keys())}")

            for at_iter in [3, bm.T_INFERENCE]:
                bm.print_table(merged, at_iter=at_iter, exp_name=exp_name, problem=problem)
                bm.print_per_N_table(merged, at_iter=at_iter)

            bm.plot_convergence(merged, exp_name=exp_name, problem=problem, fix_p=3/7)

            # Save merged pkl
            merged_path = os.path.join(rdir, f'results_{exp_name}_{problem}_merged.pkl')
            with open(merged_path, 'wb') as f:
                pickle.dump(merged, f)
            bm.log.info(f"  Saved merged: {merged_path}")

    # 4 plots: problem × experiment → saved in result/
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    for problem in ['maxcut', 'sk']:
        for exp_name in ['exp1', 'exp2']:
            _plot_single(problem, exp_name, rdir, result_dir)

    save_summary_table(rdir, result_dir)

    bm.log.info(f"\n{'='*60}")
    bm.log.info(f"Merge complete. Results saved to: {result_dir}/")
    bm.log.info("=" * 60)


if __name__ == '__main__':
    main()
