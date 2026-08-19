"""v3 on band_gap vs the original method and top baselines (Uncertainty
Sampling, QBC), same protocol as the primary benchmark."""
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

from run_primary_benchmark import run_al_trial, prepare_task_split, TRIAL_SEEDS, all_method_factories  # noqa: E402
from quantum_al.data_utils import FEATURE_COLUMNS  # noqa: E402
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
    print(f"  [{name}] MEAN R2 = {mean:.4f} +- {std:.4f}" if mean is not None else f"  [{name}] ALL FAILED")
    return {"trials": trials_out, "final_r2_mean": mean, "final_r2_std": std}


def main():
    task = "band_gap"
    out = {}

    print("--- Quantum-TreeEnsemble (v3, residual-coupled) ---")
    out["quantum_tree_ensemble"] = run_variant(
        task, "quantum_tree_ensemble",
        lambda: TreeEnsembleQuantumSelector(seed=0),
    )

    print("\n--- Reference: original quantum, Uncertainty Sampling, QBC ---")
    d = len(FEATURE_COLUMNS)
    factories = all_method_factories(d)
    for name in ["Quantum-Enhanced", "Uncertainty Sampling", "Query by Committee"]:
        out[name] = run_variant(task, name, factories[name])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "v3_test.json"), "w") as f:
        json.dump(out, f)

    print("\n=== SUMMARY (band_gap) ===")
    for name, v in out.items():
        print(f"  {name:30s} R2={v['final_r2_mean']:.4f}+-{v['final_r2_std']:.4f}")


if __name__ == "__main__":
    main()
