"""Correctness tests for quantum_al.operator_sparse: non-commutativity
still holds with sparse observables, and non-overlapping blocks give
exactly zero covariance."""
import numpy as np
import pytest

from quantum_al.operator import default_feature_groups, encode_states
from quantum_al.operator_sparse import (
    SparseQuantumObservableBank,
    SparseQuantumSelector,
    make_sparse_observable,
)


def test_sparse_observable_is_zero_outside_block():
    d = 12
    group = [2, 3, 4, 5]
    O = make_sparse_observable(d, group, seed=0)
    mask = np.ones((d, d), dtype=bool)
    mask[np.ix_(group, group)] = False
    assert np.all(O[mask] == 0.0)


def test_sparse_bank_still_noncommuting():
    d = 12
    groups = default_feature_groups(d)
    bank = SparseQuantumObservableBank(d, groups, seed=1)
    assert bank.commutator_norm() > 1e-6


def test_nonoverlapping_blocks_have_exact_zero_covariance():
    """Two sparse observables with disjoint feature support must have an
    exactly zero symmetrized covariance (not just small)."""
    d = 12
    O1 = make_sparse_observable(d, [0, 1, 2], seed=1)
    O2 = make_sparse_observable(d, [6, 7, 8], seed=2)
    sym = (O1 @ O2 + O2 @ O1) / 2.0
    assert np.max(np.abs(sym)) == 0.0


def test_sparse_selector_selects_valid_batch():
    rng = np.random.default_rng(0)
    d = 21
    X_train = rng.normal(size=(20, d))
    y_train = X_train[:, 0] + rng.normal(scale=0.1, size=20)
    X_candidates = rng.normal(size=(15, d))
    selector = SparseQuantumSelector(d, seed=0)
    n_select = 5
    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, y_train, n_select=n_select
    )
    selected_idx = np.asarray(selected_idx)
    assert len(selected_idx) == n_select
    assert selected_idx.min() >= 0
    assert selected_idx.max() < len(X_candidates)
    assert np.all(np.isfinite(scores))
