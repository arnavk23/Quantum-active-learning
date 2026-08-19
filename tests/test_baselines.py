"""
Smoke tests for the 9 classical active-learning baselines
(quantum_al.baselines). Uses small synthetic arrays, not real Materials
Project data (which needs a live API key), so these run in any CI
environment without network access. The goal is only to verify each
selector returns a valid selection on a toy regression problem, not to
reproduce any paper result.
"""
import numpy as np
import pytest

from quantum_al.baselines import BaselineFactory, BASELINE_METHODS, get_all_baselines


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 6))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + rng.normal(scale=0.1, size=20)
    X_candidates = rng.normal(size=(15, 6))
    return X_candidates, X_train, y_train


def test_get_all_baselines_returns_nine():
    baselines = get_all_baselines()
    assert len(baselines) == 9
    assert set(baselines.keys()) == {name for name, _ in BASELINE_METHODS}


@pytest.mark.parametrize("name,method_attr", BASELINE_METHODS)
def test_baseline_selects_valid_batch(name, method_attr, toy_data):
    X_candidates, X_train, y_train = toy_data
    selector = getattr(BaselineFactory(), method_attr)()
    n_select = 5
    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, y_train, n_select=n_select
    )
    selected_idx = np.asarray(selected_idx)
    assert len(selected_idx) == n_select
    assert selected_idx.min() >= 0
    assert selected_idx.max() < len(X_candidates)
    assert len(set(selected_idx.tolist())) == n_select  # no duplicates


@pytest.mark.parametrize("name,method_attr", BASELINE_METHODS)
def test_baseline_handles_large_n_select(name, method_attr, toy_data):
    """The benchmark harness (benchmarks/run_primary_benchmark.py) relies
    on every selector tolerating n_select > len(X_candidates) without
    raising, clamping to the available pool instead."""
    X_candidates, X_train, y_train = toy_data
    selector = getattr(BaselineFactory(), method_attr)()
    n_select = len(X_candidates) + 10
    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, y_train, n_select=n_select
    )
    selected_idx = np.asarray(selected_idx)
    assert len(selected_idx) == len(X_candidates)
    assert selected_idx.min() >= 0
    assert selected_idx.max() < len(X_candidates)
    assert len(set(selected_idx.tolist())) == len(X_candidates)
