"""
Tests the two principled improvements in quantum_operator_v2.py against the
original faithful implementation, on the same protocol as the primary
benchmark. Reports whatever actually happens -- this is a single,
pre-specified round of improvement, not an iterate-until-it-wins search.
"""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)

from quantum_al.data_utils import load_task, standardize, FEATURE_COLUMNS, REGRESSION_TASKS  # noqa: E402
from run_primary_benchmark import run_al_trial, prepare_task_split, N0, T_ITERS, BATCH_SIZE, TRIAL_SEEDS  # noqa: E402
from quantum_al.operator import default_feature_groups  # noqa: E402
from quantum_al.operator_v2 import (  # noqa: E402
    domain_feature_groups, ImportanceWeightedQuantumSelector,
)
import quantum_al.operator as qo  # noqa: E402


class PlainQuantumDomainGroups:
    """Original QuantumSelector logic, but with domain-informed feature
    groups instead of arbitrary contiguous thirds. Isolates the effect of
    (1) alone, without (2)'s importance weighting."""
    def __init__(self, d, seed=0, name="Quantum-DomainGroups"):
        self.name = name
        groups = domain_feature_groups()
        self.bank = qo.QuantumObservableBank(d, groups, seed=seed)

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        scores = self.bank.batch_scores(X_train, X_candidates, use_covariance=True)
        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {"quantum_scores": scores}


class PlainQuantumOriginal:
    def __init__(self, d, seed=0, name="Quantum-Original"):
        self.name = name
        groups = default_feature_groups(d)
        self.bank = qo.QuantumObservableBank(d, groups, seed=seed)

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        scores = self.bank.batch_scores(X_train, X_candidates, use_covariance=True)
        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {"quantum_scores": scores}


VARIANTS = {
    "original_default_thirds": lambda d: PlainQuantumOriginal(d, seed=0),
    "domain_groups_only": lambda d: PlainQuantumDomainGroups(d, seed=0),
    "domain_groups_plus_importance_weighting": lambda d: ImportanceWeightedQuantumSelector(d, seed=0),
}


def run_on_task(task):
    print(f"\n{'='*70}\ntask: {task}\n{'='*70}")
    d = len(FEATURE_COLUMNS)
    out = {}
    for vname, ctor in VARIANTS.items():
        print(f"\n--- variant: {vname} ---")
        trials_out = []
        for trial in TRIAL_SEEDS:
            X_pool, y_pool, X_test, y_test = prepare_task_split(task, trial)

            def factory(ctor=ctor, d_=d):
                return ctor(d_)

            result = run_al_trial(factory, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                                   method_seed_offset=0)
            trials_out.append(result)
            final_r2 = result["r2"][-1] if result["r2"] else float("nan")
            print(f"  trial={trial} final R2={final_r2:.4f} error={result['error']}")
        finals = [t["r2"][-1] for t in trials_out if not t["error"]]
        out[vname] = {
            "trials": trials_out,
            "final_r2_mean": float(np.mean(finals)) if finals else None,
            "final_r2_std": float(np.std(finals)) if finals else None,
        }
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Stage 1: quick feedback on band_gap only.
    results = {"band_gap": run_on_task("band_gap")}
    with open(os.path.join(RESULTS_DIR, "improvement_attempt.json"), "w") as f:
        json.dump(results, f)

    print("\n\n=== SUMMARY (band_gap) ===")
    for vname, v in results["band_gap"].items():
        print(f"  {vname:40s} R2={v['final_r2_mean']:.4f}+-{v['final_r2_std']:.4f}")


if __name__ == "__main__":
    main()
