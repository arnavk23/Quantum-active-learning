"""Smoke tests for the quantum-inspired selectors used in the improvement
attempt and the residual-coupled fix (quantum_al.operator_v2,
quantum_al.operator_v3), mirroring tests/test_baselines.py's coverage of
the classical baselines."""
import numpy as np
import pytest

from quantum_al.data_utils import FEATURE_COLUMNS
from quantum_al.operator_v2 import (
    ImportanceWeightedQuantumSelector,
    domain_feature_groups,
)
from quantum_al.operator_v3 import TreeEnsembleQuantumSelector


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    d = len(FEATURE_COLUMNS)
    X_train = rng.normal(size=(20, d))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + rng.normal(scale=0.1, size=20)
    X_candidates = rng.normal(size=(15, d))
    return X_candidates, X_train, y_train, d


def _assert_valid_selection(selected_idx, scores, n_candidates, n_select):
    selected_idx = np.asarray(selected_idx)
    assert len(selected_idx) == n_select
    assert selected_idx.min() >= 0
    assert selected_idx.max() < n_candidates
    assert len(set(selected_idx.tolist())) == n_select  # no duplicates
    assert np.all(np.isfinite(scores))


def test_domain_feature_groups_cover_all_columns():
    groups = domain_feature_groups()
    all_indices = sorted(i for idxs in groups.values() for i in idxs)
    assert all_indices == list(range(len(FEATURE_COLUMNS)))


def test_importance_weighted_selector_selects_valid_batch(toy_data):
    X_candidates, X_train, y_train, d = toy_data
    selector = ImportanceWeightedQuantumSelector(d, seed=0)
    n_select = 5
    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, y_train, n_select=n_select
    )
    _assert_valid_selection(selected_idx, scores, len(X_candidates), n_select)
    assert "importance_weights" in info
    assert len(info["importance_weights"]) == d


def test_tree_ensemble_selector_selects_valid_batch(toy_data):
    X_candidates, X_train, y_train, _ = toy_data
    selector = TreeEnsembleQuantumSelector(seed=0, n_estimators=9)
    n_select = 5
    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, y_train, n_select=n_select
    )
    _assert_valid_selection(selected_idx, scores, len(X_candidates), n_select)


@pytest.mark.parametrize("use_covariance", [True, False])
def test_tree_ensemble_selector_covariance_toggle(toy_data, use_covariance):
    """Both the full and no-covariance ablation variants used in
    results/v3_ablation.json should produce valid, finite scores."""
    X_candidates, X_train, y_train, _ = toy_data
    selector = TreeEnsembleQuantumSelector(seed=0, n_estimators=9, use_covariance=use_covariance)
    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, y_train, n_select=5
    )
    _assert_valid_selection(selected_idx, scores, len(X_candidates), 5)
