"""Sparse-by-construction observables vs dense+grouped: Pauli overhead and
real-data accuracy, on the original formalism and the residual-coupled
(v3) fix. Saves results/sparse_observable_experiment.json.
"""
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)

from run_primary_benchmark import run_al_trial, prepare_task_split, TRIAL_SEEDS  # noqa: E402
from quantum_al.data_utils import FEATURE_COLUMNS  # noqa: E402
from quantum_al.operator import default_feature_groups, encode_states, feature_phase_weights  # noqa: E402
from quantum_al.operator_sparse import SparseQuantumObservableBank, SparseQuantumSelector  # noqa: E402
from quantum_al.operator_v3 import tree_group_split  # noqa: E402
from quantum_al.circuit import pad_observable, pauli_decompose, n_qubits_for_dim  # noqa: E402


def safe_qwc_count(op):
    return 0 if len(op.paulis) == 0 else len(op.group_commuting(qubit_wise=True))


def pauli_overhead(make_observable_fn, d, groups, seed=1):
    from quantum_al.operator import make_observable as dense_make_observable
    names = list(groups.keys())
    ctor = make_observable_fn or dense_make_observable
    nq = n_qubits_for_dim(d)
    O = {n: ctor(d, groups[n], seed=seed + i) for i, n in enumerate(names)}
    O_pad = {n: pad_observable(O[n], nq) for n in names}
    total_raw = total_qwc = 0
    per_quantity = {}
    for n in names:
        for label, mat in [("O", O_pad[n]), ("O2", O_pad[n] @ O_pad[n])]:
            op = pauli_decompose(mat)
            qwc = safe_qwc_count(op)
            per_quantity[f"{n}_{label}"] = {"raw": len(op.paulis), "qwc": qwc}
            total_raw += len(op.paulis)
            total_qwc += qwc
    for i, k in enumerate(names):
        for l in names[i + 1:]:
            sym = (O_pad[k] @ O_pad[l] + O_pad[l] @ O_pad[k]) / 2.0
            op = pauli_decompose(sym)
            qwc = safe_qwc_count(op)
            per_quantity[f"{k}_{l}_sym"] = {"raw": len(op.paulis), "qwc": qwc}
            total_raw += len(op.paulis)
            total_qwc += qwc
    return {"n_qubits": nq, "total_raw": total_raw, "total_qwc": total_qwc, "per_quantity": per_quantity}


class SparseTreeEnsembleSelector:
    def __init__(self, seed=0, n_estimators=99, n_groups=3, use_covariance=True):
        self.seed = seed
        self.n_estimators = n_estimators
        self.n_groups = n_groups
        self.use_covariance = use_covariance

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.seed)
        rf.fit(X_train, y_train)
        tree_preds_cand = np.stack([t.predict(X_candidates) for t in rf.estimators_], axis=1)
        tree_preds_train = np.stack([t.predict(X_train) for t in rf.estimators_], axis=1)
        mu = tree_preds_train.mean(axis=0)
        sigma = tree_preds_train.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        Z_train = (tree_preds_train - mu) / sigma
        Z_cand = (tree_preds_cand - mu) / sigma
        n_trees = Z_cand.shape[1]
        groups = tree_group_split(n_trees, self.n_groups)
        bank = SparseQuantumObservableBank(n_trees, groups, seed=self.seed)
        phase_weights = feature_phase_weights(Z_train)
        psi_cand = encode_states(Z_cand, phase_weights)
        scores = np.array([
            bank.total_uncertainty(psi_cand[i], use_covariance=self.use_covariance)
            for i in range(psi_cand.shape[0])
        ])
        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {}


def run_variant_on_band_gap(factory_fn):
    finals = []
    for trial in TRIAL_SEEDS:
        X_pool, y_pool, X_test, y_test = prepare_task_split("band_gap", trial)
        result = run_al_trial(factory_fn, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                               method_seed_offset=0)
        finals.append(result["r2"][-1] if result["r2"] else float("nan"))
    return finals


def main():
    d = len(FEATURE_COLUMNS)
    groups = default_feature_groups(d)

    print("Pauli overhead, K=3 raw features (d=21) ...")
    from quantum_al.operator_sparse import make_sparse_observable
    from quantum_al.operator import make_observable
    dense_overhead = pauli_overhead(make_observable, d, groups)
    sparse_overhead = pauli_overhead(make_sparse_observable, d, groups)
    print("dense:", dense_overhead["total_raw"], dense_overhead["total_qwc"])
    print("sparse:", sparse_overhead["total_raw"], sparse_overhead["total_qwc"])

    print("\nAccuracy: sparse vs dense, original formalism, band_gap ...")
    sparse_finals = run_variant_on_band_gap(lambda: SparseQuantumSelector(d, seed=0))
    with open(os.path.join(RESULTS_DIR, "primary_benchmark_summary.json")) as f:
        dense_finals = json.load(f)["band_gap"]["Quantum-Enhanced"]["final_r2_per_trial"]
    t_orig, p_orig = stats.ttest_rel(sparse_finals, dense_finals)
    print("sparse:", sparse_finals, "dense:", dense_finals, f"t={t_orig:.3f} p={p_orig:.4f}")

    print("\nAccuracy: sparse vs dense, v3 residual-coupled fix, band_gap ...")
    sparse_v3_finals = run_variant_on_band_gap(
        lambda: SparseTreeEnsembleSelector(seed=0, n_estimators=99)
    )
    with open(os.path.join(RESULTS_DIR, "v3_all_tasks.json")) as f:
        dense_v3_full = json.load(f)
    dense_v3_finals = [t["r2"][-1] for t in dense_v3_full["band_gap"]["v3_full"]["trials"] if not t["error"]]
    t_v3, p_v3 = stats.ttest_rel(sparse_v3_finals, dense_v3_finals)
    print("sparse v3:", sparse_v3_finals, "dense v3:", dense_v3_finals, f"t={t_v3:.3f} p={p_v3:.4f}")

    print("\nVerifying exact-zero covariance for non-overlapping tree groups ...")
    n_trees = 99
    tgroups = tree_group_split(n_trees, 3)
    tnames = list(tgroups.keys())
    O_tree = {n: make_sparse_observable(n_trees, tgroups[n], seed=1 + i) for i, n in enumerate(tnames)}
    max_cov_entries = {}
    for i, k in enumerate(tnames):
        for l in tnames[i + 1:]:
            sym = (O_tree[k] @ O_tree[l] + O_tree[l] @ O_tree[k]) / 2.0
            max_cov_entries[f"{k}_{l}"] = float(np.max(np.abs(sym)))
    print(max_cov_entries)

    out = {
        "pauli_overhead_K3_d21": {"dense": dense_overhead, "sparse": sparse_overhead},
        "accuracy_original_formalism_band_gap": {
            "sparse_per_trial": sparse_finals, "dense_per_trial": dense_finals,
            "t_statistic": float(t_orig), "p_value": float(p_orig),
        },
        "accuracy_v3_band_gap": {
            "sparse_per_trial": sparse_v3_finals, "dense_per_trial": dense_v3_finals,
            "t_statistic": float(t_v3), "p_value": float(p_v3),
        },
        "v3_tree_group_covariance_exact_zero_check": max_cov_entries,
    }
    with open(os.path.join(RESULTS_DIR, "sparse_observable_experiment.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/sparse_observable_experiment.json")


if __name__ == "__main__":
    main()
