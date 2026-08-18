"""
Residual/disagreement-coupled quantum-inspired acquisition (v3).

Directly targets the diagnosis from quantum_operator_v2.py's failed
improvement attempt: U_total was a function only of a candidate's static
position in the raw standardized feature space, never of what the current
predictor actually finds uncertain. v2's two fixes (domain-informed
observable groups, importance-weighted feature rescaling) kept that
structural problem and, unsurprisingly, both failed.

This module removes the problem at its source: instead of encoding
|psi_i> from candidate M_i's raw material features, it encodes |psi_i>
from the vector of per-tree predictions of a RandomForest fit on the
*current* labeled set (X_train, y_train) -- i.e. exactly the ensemble-
disagreement signal that Query-by-Committee and RF-Uncertainty already
win with. The trees are split into 3 arbitrary equal-size groups
("ensemble observables" -- there is no physical structural/electronic/
thermodynamic meaning once the basis is tree predictions rather than
material features) and the existing, unmodified covariance-aware
aggregation machinery (quantum_operator.QuantumObservableBank) is run on
top of that basis exactly as before.

This is the fair test of the paper's actual scientific question: does
covariance-aware, complex-coupled aggregation across non-commuting
observables add value *on top of* ensemble-disagreement information,
compared to just taking the disagreement's raw variance (= RF
Uncertainty baseline)? If this variant beats RF Uncertainty and QBC, that
is genuine evidence for the covariance-coupling hypothesis. If it does
not, that is a fair, no-longer-confounded negative result -- the earlier
failure could no longer be blamed on the acquisition score's blindness to
model disagreement, because this variant is not blind to it.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from quantum_al.operator import QuantumObservableBank, encode_states, feature_phase_weights


def tree_group_split(n_trees, k=3):
    """Split n_trees indices into k equal-ish contiguous groups (no
    semantic meaning across groups -- trees are exchangeable)."""
    block = n_trees // k
    groups = {}
    for i in range(k):
        start = i * block
        end = n_trees if i == k - 1 else (i + 1) * block
        groups[f"tree_group_{i}"] = list(range(start, end))
    return groups


class TreeEnsembleQuantumSelector:
    """Covariance-aware quantum aggregation over per-tree RF predictions
    instead of raw material features -- couples U_total to genuine model
    disagreement while preserving the covariance/complex-coefficient
    machinery being tested."""

    def __init__(self, seed=0, n_estimators=99, n_groups=3,
                 use_covariance=True, name="Quantum-TreeEnsemble"):
        self.name = name
        self.seed = seed
        self.n_estimators = n_estimators  # divisible by n_groups=3 for clean split
        self.n_groups = n_groups
        self.use_covariance = use_covariance

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.seed)
        rf.fit(X_train, y_train)

        # Per-tree predictions: (n_candidates, n_trees) and (n_train, n_trees)
        tree_preds_cand = np.stack([t.predict(X_candidates) for t in rf.estimators_], axis=1)
        tree_preds_train = np.stack([t.predict(X_train) for t in rf.estimators_], axis=1)

        # Standardize per-tree-prediction "features" using the training pool's
        # own statistics (same convention as the raw-feature pipeline).
        mu = tree_preds_train.mean(axis=0)
        sigma = tree_preds_train.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        Z_train = (tree_preds_train - mu) / sigma
        Z_cand = (tree_preds_cand - mu) / sigma

        n_trees = Z_cand.shape[1]
        groups = tree_group_split(n_trees, self.n_groups)
        bank = QuantumObservableBank(n_trees, groups, seed=self.seed)

        phase_weights = feature_phase_weights(Z_train)
        psi_cand = encode_states(Z_cand, phase_weights)
        scores = np.array([
            bank.total_uncertainty(psi_cand[i], use_covariance=self.use_covariance)
            for i in range(psi_cand.shape[0])
        ])

        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {"quantum_scores": scores}
