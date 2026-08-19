"""State encoded from per-tree RF predictions instead of raw features, so
U_total is coupled to ensemble disagreement (results/v3_*.json). Same
QuantumObservableBank aggregation on top, unmodified. Trees split into
n_groups exchangeable buckets, no physical meaning per group.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from quantum_al.operator import QuantumObservableBank, encode_states, feature_phase_weights


def tree_group_split(n_trees, k=3):
    """Split n_trees indices into k equal-ish contiguous groups."""
    block = n_trees // k
    groups = {}
    for i in range(k):
        start = i * block
        end = n_trees if i == k - 1 else (i + 1) * block
        groups[f"tree_group_{i}"] = list(range(start, end))
    return groups


class TreeEnsembleQuantumSelector:
    """U_total ranking over per-tree RF predictions instead of raw features."""

    def __init__(self, seed=0, n_estimators=99, n_groups=3,
                 use_covariance=True, name="Quantum-TreeEnsemble"):
        self.name = name
        self.seed = seed
        self.n_estimators = n_estimators  # divisible by n_groups for a clean split
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
