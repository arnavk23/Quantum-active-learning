"""Observables sparse by construction (nonzero only within their own
feature block) instead of dense-then-measurement-grouped. Tests whether
this cuts the NISQ Pauli-term overhead further than grouping alone.
"""
import numpy as np

from quantum_al.operator import QuantumObservableBank


def make_sparse_observable(d, feature_group, seed):
    """Hermitian matrix, zero outside feature_group x feature_group."""
    rng = np.random.default_rng(seed)
    k = len(feature_group)
    A_local = rng.normal(size=(k, k))
    H_local = (A_local + A_local.T) / 2.0
    idx = np.array(feature_group)
    H = np.zeros((d, d))
    H[np.ix_(idx, idx)] = H_local
    eigmax = np.max(np.abs(np.linalg.eigvalsh(H)))
    if eigmax > 1e-12:
        H = H / eigmax
    return H


class SparseQuantumObservableBank(QuantumObservableBank):
    """Same interface as QuantumObservableBank, sparse observables."""

    def __init__(self, d, feature_groups, coefficients=None, seed=0):
        super().__init__(d, feature_groups, coefficients=coefficients, seed=seed)
        self.O = {
            name: make_sparse_observable(d, feature_groups[name], seed=seed + i)
            for i, name in enumerate(self.names)
        }


class SparseQuantumSelector:
    """batch_scores-ranking selector using SparseQuantumObservableBank."""

    def __init__(self, d, feature_groups=None, seed=0, use_covariance=True,
                 name="Quantum-Sparse"):
        from quantum_al.operator import default_feature_groups
        self.name = name
        groups = feature_groups if feature_groups is not None else default_feature_groups(d)
        self.bank = SparseQuantumObservableBank(d, groups, seed=seed)
        self.use_covariance = use_covariance

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        scores = self.bank.batch_scores(X_train, X_candidates, use_covariance=self.use_covariance)
        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {"quantum_scores": scores}
