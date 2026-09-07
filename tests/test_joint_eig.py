"""Correctness tests for quantum_al.joint_eig: the Hadamard-inequality
decomposition (marginal_sum - joint = total_correlation_gap >= 0) holds
exactly, collapses to zero for uncorrelated tasks, and the selectors
produce valid batches on real-shaped multi-output data.
"""
import numpy as np
import pytest

from quantum_al.joint_eig import (
    self_test,
    joint_eig_score,
    marginal_sum_score,
    total_correlation_gap,
    JointEIGSelector,
    MarginalSumSelector,
)


def test_self_test_passes():
    assert self_test()


def test_gap_matches_logdet_correlation_identity():
    rng = np.random.default_rng(3)
    n, K = 10, 4
    A = rng.normal(size=(n, K, K))
    Sigma = np.einsum("nij,nkj->nik", A, A) + 1e-3 * np.eye(K)[None]
    R = np.abs(rng.normal(size=K)) + 1e-2

    gap = total_correlation_gap(Sigma, R)
    assert np.all(gap >= -1e-9)

    total = Sigma + np.diag(R)[None]
    d = np.diagonal(total, axis1=1, axis2=2)
    corr = total / np.sqrt(d[:, :, None] * d[:, None, :])
    _, logdet_corr = np.linalg.slogdet(corr)
    assert np.allclose(gap, -0.5 * logdet_corr, atol=1e-8)


def test_gap_is_zero_for_diagonal_covariance():
    K = 3
    Sigma = np.diag([1.0, 2.0, 0.5])[None]
    R = np.array([0.1, 0.1, 0.1])
    gap = total_correlation_gap(Sigma, R)
    assert np.allclose(gap, 0.0, atol=1e-10)


def test_joint_selector_selects_valid_batch():
    rng = np.random.default_rng(0)
    d, K = 21, 2
    X_train = rng.normal(size=(60, d))
    Y_train = np.stack([
        X_train[:, 0] + rng.normal(scale=0.1, size=60),
        -0.5 * X_train[:, 0] + X_train[:, 1] + rng.normal(scale=0.1, size=60),
    ], axis=1)
    X_candidates = rng.normal(size=(15, d))

    selector = JointEIGSelector(n_estimators=50, seed=0)
    n_select = 5
    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, Y_train, n_select=n_select
    )
    selected_idx = np.asarray(selected_idx)
    assert len(selected_idx) == n_select
    assert selected_idx.min() >= 0
    assert selected_idx.max() < len(X_candidates)
    assert np.all(np.isfinite(scores))
    assert "forest" in info and "R" in info


def test_marginal_sum_selector_selects_valid_batch():
    rng = np.random.default_rng(1)
    d, K = 21, 2
    X_train = rng.normal(size=(60, d))
    Y_train = rng.normal(size=(60, K))
    X_candidates = rng.normal(size=(15, d))

    selector = MarginalSumSelector(n_estimators=50, seed=0)
    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, Y_train, n_select=4
    )
    assert len(np.asarray(selected_idx)) == 4
    assert np.all(np.isfinite(scores))
