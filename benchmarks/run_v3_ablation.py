"""Ablation on v3: does covariance aggregation add anything over plain
ensemble disagreement?"""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)  # only for sibling benchmarks/*.py scripts below,
# which are standalone (not part of the installed quantum_al package);
# quantum_al.* imports themselves work without this once pip-installed.

from run_primary_benchmark import run_al_trial, prepare_task_split, TRIAL_SEEDS  # noqa: E402
from quantum_al.operator_v3 import TreeEnsembleQuantumSelector  # noqa: E402


def run_variant(task, name, factory_fn):
    trials_out = []
    for trial in TRIAL_SEEDS:
        X_pool, y_pool, X_test, y_test = prepare_task_split(task, trial)
        result = run_al_trial(factory_fn, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                               method_seed_offset=0)
        trials_out.append(result)
        final_r2 = result["r2"][-1] if result["r2"] else float("nan")
        print(f"  [{name}] trial={trial} final R2={final_r2:.4f} error={result['error']}")
    finals = [t["r2"][-1] for t in trials_out if not t["error"]]
    mean, std = (float(np.mean(finals)), float(np.std(finals))) if finals else (None, None)
    print(f"  [{name}] MEAN R2 = {mean:.4f} +- {std:.4f}")
    return {"trials": trials_out, "final_r2_mean": mean, "final_r2_std": std}


def main():
    task = "band_gap"
    out = {}
    print("--- v3 full (with covariance) ---")
    out["v3_with_covariance"] = run_variant(
        task, "v3_with_covariance", lambda: TreeEnsembleQuantumSelector(seed=0, use_covariance=True))

    print("\n--- v3 no covariance (Cov=0, plain weighted variance sum over tree-groups) ---")
    out["v3_no_covariance"] = run_variant(
        task, "v3_no_covariance", lambda: TreeEnsembleQuantumSelector(seed=0, use_covariance=False))

    print("\n--- v3 single group (K=1, ~ plain per-candidate tree-prediction variance, like RF Uncertainty) ---")
    out["v3_single_group"] = run_variant(
        task, "v3_single_group", lambda: TreeEnsembleQuantumSelector(seed=0, use_covariance=True, n_groups=1))

    with open(os.path.join(RESULTS_DIR, "v3_ablation.json"), "w") as f:
        json.dump(out, f)

    print("\n=== SUMMARY ===")
    for name, v in out.items():
        print(f"  {name:20s} R2={v['final_r2_mean']:.4f}+-{v['final_r2_std']:.4f}")


if __name__ == "__main__":
    main()
