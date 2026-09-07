"""Tests the joint-EIG acquisition function (quantum_al.joint_eig) against
its own classical-limit ablation (marginal-sum) and random sampling, on
two real, jointly-labeled MP properties with a genuine, non-trivial
correlation (band_gap, formation_energy: r=-0.365, n=498 shared materials).

Unlike the original single-property comparisons in run_primary_benchmark.py,
this experiment's setting is: one AL query returns BOTH labels for the
chosen material (as a real DFT run would), so a genuinely joint acquisition
score is a meaningful, distinct question from per-property scalar
uncertainty. Protocol constants match run_primary_benchmark.py for
consistency (N0=50, T=8, batch=15, 5 trials, 30% held-out test set).

Usage: python benchmarks/run_joint_eig_experiment.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)

from quantum_al.data_utils import load_multi_task, standardize  # noqa: E402
from quantum_al.joint_eig import (  # noqa: E402
    JointEIGSelector, MarginalSumSelector,
    predictive_covariance, oob_residual_variance, total_correlation_gap,
)

os.makedirs(RESULTS_DIR, exist_ok=True)

N0_DEFAULT = 50
T_ITERS_DEFAULT = 8
BATCH_SIZE_DEFAULT = 15
TEST_FRACTION = 0.3
N_ESTIMATORS = 200


class RandomSelector:
    name = "Random"

    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        n_select = min(n_select, len(X_candidates))
        idx = np.random.permutation(len(X_candidates))[:n_select]
        return idx, np.zeros(len(X_candidates)), {}


def holm_bonferroni(pvals, alpha=0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = pvals[idx] * factor
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    reject = adj < alpha
    return adj.tolist(), reject.tolist()


def run_al_trial(method_factory, X_pool, Y_pool, X_test, Y_test, trial_seed,
                  n0=N0_DEFAULT, t_iters=T_ITERS_DEFAULT, batch_size=BATCH_SIZE_DEFAULT,
                  method_seed_offset=0, track_gap=False):
    rng = np.random.RandomState(trial_seed)
    n_pool = X_pool.shape[0]
    perm = rng.permutation(n_pool)
    labeled_idx = perm[:n0].tolist()
    remaining = perm[n0:].tolist()

    np.random.seed(trial_seed * 10007 + method_seed_offset)
    method = method_factory()

    iterations, r2_joint, r2_per_task, n_labeled_list, gaps = [], [], [], [], []
    error = None
    try:
        from sklearn.ensemble import RandomForestRegressor
        for it in range(t_iters + 1):
            X_train = X_pool[labeled_idx]
            Y_train = Y_pool[labeled_idx]
            rf = RandomForestRegressor(n_estimators=100, random_state=trial_seed, n_jobs=-1)
            rf.fit(X_train, Y_train)
            pred = rf.predict(X_test)
            per_task = r2_score(Y_test, pred, multioutput="raw_values")
            joint_r2 = float(np.mean(per_task))

            iterations.append(it)
            r2_joint.append(joint_r2)
            r2_per_task.append([float(v) for v in per_task])
            n_labeled_list.append(len(labeled_idx))

            if it == t_iters or len(remaining) == 0:
                break

            X_candidates = X_pool[remaining]
            sel_idx, scores, info = method.select_next_experiments(
                X_candidates, X_train, Y_train, n_select=min(batch_size, len(remaining))
            )

            if track_gap and "forest" in info:
                Sigma = predictive_covariance(info["forest"], X_candidates)
                gap = total_correlation_gap(Sigma, info["R"])
                gaps.append(float(np.mean(gap)))

            sel_idx = np.asarray(sel_idx).astype(int)
            chosen_global = [remaining[i] for i in sel_idx]
            labeled_idx.extend(chosen_global)
            remaining = [i for i in remaining if i not in set(chosen_global)]
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return {
        "trial_seed": trial_seed,
        "iteration": iterations,
        "r2_joint": r2_joint,
        "r2_per_task": r2_per_task,
        "n_labeled": n_labeled_list,
        "mean_total_correlation_gap": gaps,
        "error": error,
    }


def main(tasks, n_trials, out_name, n0=N0_DEFAULT, t_iters=T_ITERS_DEFAULT, batch_size=BATCH_SIZE_DEFAULT):
    trial_seeds = list(range(n_trials))
    print("=" * 70)
    print(f"Joint-EIG experiment: {tasks[0]} + {tasks[1]} (n_trials={n_trials}, "
          f"n0={n0}, T={t_iters}, batch={batch_size})")
    print("=" * 70)
    X, Y, meta = load_multi_task(tasks)
    print(f"n_materials={X.shape[0]} n_features={X.shape[1]} n_tasks={Y.shape[1]}")
    pair_corr = float(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1])
    print(f"raw label correlation r={pair_corr:.3f}")

    factories = {
        "Joint-EIG": lambda: JointEIGSelector(n_estimators=N_ESTIMATORS, seed=0),
        "Marginal-Sum": lambda: MarginalSumSelector(n_estimators=N_ESTIMATORS, seed=0),
        "Random": lambda: RandomSelector(),
    }
    seed_offsets = {"Joint-EIG": 0, "Marginal-Sum": 1, "Random": 2}

    out = {
        "config": {
            "tasks": tasks, "n0": n0, "T": t_iters, "batch_size": batch_size,
            "n_trials": n_trials, "test_fraction": TEST_FRACTION,
            "n_estimators": N_ESTIMATORS, "raw_label_correlation": pair_corr,
            "note": "one query returns labels for BOTH tasks (joint DFT-run setting)",
        },
        "methods": {},
    }

    for name, factory in factories.items():
        out["methods"][name] = {"trials": []}
        for trial in trial_seeds:
            X_pool_raw, X_test_raw, Y_pool, Y_test = train_test_split(
                X, Y, test_size=TEST_FRACTION, random_state=trial
            )
            X_pool, X_test = standardize(X_pool_raw, X_test_raw)
            t0 = time.time()
            result = run_al_trial(
                factory, X_pool, Y_pool, X_test, Y_test, trial_seed=trial,
                n0=n0, t_iters=t_iters, batch_size=batch_size,
                method_seed_offset=seed_offsets[name],
                track_gap=(name == "Joint-EIG"),
            )
            dt = time.time() - t0
            status = "ERROR" if result["error"] else "ok"
            final = result["r2_joint"][-1] if result["r2_joint"] else float("nan")
            print(f"  {name:14s} trial={trial} final joint-R2={final:.4f} ({dt:.1f}s) {status}")
            out["methods"][name]["trials"].append(result)

    summary = {}
    for name in out["methods"]:
        trials = out["methods"][name]["trials"]
        finals = [t["r2_joint"][-1] for t in trials if not t["error"]]
        finals_per_task = [t["r2_per_task"][-1] for t in trials if not t["error"]]
        finals_per_task = np.array(finals_per_task) if finals_per_task else np.zeros((0, len(tasks)))
        summary[name] = {
            "final_joint_r2_mean": float(np.mean(finals)) if finals else None,
            "final_joint_r2_std": float(np.std(finals)) if finals else None,
            "final_joint_r2_per_trial": finals,
            "final_per_task_r2_mean": finals_per_task.mean(axis=0).tolist() if len(finals_per_task) else None,
        }
    out["summary"] = summary

    # paired stats: Joint-EIG vs Marginal-Sum, Joint-EIG vs Random
    jeig = np.array(summary["Joint-EIG"]["final_joint_r2_per_trial"])
    comparisons = {}
    raw_pvals, names_in_order = [], []
    for name in ["Marginal-Sum", "Random"]:
        base = np.array(summary[name]["final_joint_r2_per_trial"])
        n = min(len(jeig), len(base))
        diff = jeig[:n] - base[:n]
        t_stat, p_val = stats.ttest_rel(jeig[:n], base[:n])
        try:
            shapiro_stat, shapiro_p = stats.shapiro(diff) if n >= 3 else (None, None)
        except Exception:
            shapiro_stat, shapiro_p = (None, None)
        comparisons[name] = {
            "n_paired": n,
            "joint_eig_mean": float(np.mean(jeig[:n])),
            "baseline_mean": float(np.mean(base[:n])),
            "mean_diff": float(np.mean(diff)),
            "t_statistic": float(t_stat),
            "p_value_raw": float(p_val),
            "shapiro_p": float(shapiro_p) if shapiro_p is not None else None,
            "joint_eig_wins": bool(np.mean(diff) > 0),
        }
        raw_pvals.append(p_val)
        names_in_order.append(name)

    adj, reject = holm_bonferroni(raw_pvals, alpha=0.05)
    for name, p_adj, rej in zip(names_in_order, adj, reject):
        comparisons[name]["p_value_holm_bonferroni"] = float(p_adj)
        comparisons[name]["significant_at_0.05_after_correction"] = bool(rej)
    out["comparisons"] = comparisons

    # evidence the theorem's gap term is real and non-trivial on this data
    jeig_trials = out["methods"]["Joint-EIG"]["trials"]
    all_gaps = [g for t in jeig_trials for g in t["mean_total_correlation_gap"]]
    out["total_correlation_gap_stats"] = {
        "mean": float(np.mean(all_gaps)) if all_gaps else None,
        "min": float(np.min(all_gaps)) if all_gaps else None,
        "max": float(np.max(all_gaps)) if all_gaps else None,
        "note": "mean over candidates per AL round; always >=0 (Hadamard); "
                ">0 confirms the two tasks' epistemic uncertainties are "
                "genuinely correlated across the ensemble, not just their labels",
    }

    with open(os.path.join(RESULTS_DIR, out_name), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY (final joint R^2, mean of band_gap & formation_energy R^2)")
    print("=" * 70)
    for name, s in summary.items():
        print(f"  {name:14s} {s['final_joint_r2_mean']:.4f} +/- {s['final_joint_r2_std']:.4f}")
    print("\nComparisons vs Joint-EIG:")
    for name, c in comparisons.items():
        print(f"  vs {name:14s} mean_diff={c['mean_diff']:+.4f} p_raw={c['p_value_raw']:.4f} "
              f"p_holm={c['p_value_holm_bonferroni']:.4f} wins={c['joint_eig_wins']}")
    print(f"\nTotal-correlation gap (mean/min/max): "
          f"{out['total_correlation_gap_stats']['mean']:.4f} / "
          f"{out['total_correlation_gap_stats']['min']:.4f} / "
          f"{out['total_correlation_gap_stats']['max']:.4f}")
    print(f"\nSaved results/{out_name}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs=2, default=["band_gap", "formation_energy"])
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--out", default="joint_eig_experiment.json")
    parser.add_argument("--n0", type=int, default=N0_DEFAULT)
    parser.add_argument("--t-iters", type=int, default=T_ITERS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    args = parser.parse_args()
    main(args.tasks, args.n_trials, args.out, n0=args.n0, t_iters=args.t_iters, batch_size=args.batch_size)
