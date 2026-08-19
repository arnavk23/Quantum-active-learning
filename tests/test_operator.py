"""
Correctness tests for the core covariance-aware quantum-inspired
formalism (quantum_al.operator). These wrap the same checks the module's
own self_test() performs, as proper pytest assertions: non-commutativity
of the default observable bank, and the exact classical-limit reduction
(Proposition 2 in the papers) when observables commute, coefficients are
real, and covariance is dropped.
"""
import numpy as np
import pytest

from quantum_al.operator import (
    QuantumObservableBank,
    default_feature_groups,
    encode_states,
    feature_phase_weights,
    softmax,
)


def test_softmax_sums_to_one():
    x = np.random.default_rng(0).normal(size=(5, 8))
    p = softmax(x, axis=1)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)


def test_softmax_numerically_stable_on_large_magnitude_inputs():
    """Guards against regressions in the x - x.max() stabilization: large
    positive/negative inputs should not produce NaN/inf, should still sum
    to 1, and should concentrate probability on the largest entries."""
    x = np.array([
        [1e3, -1e3, 0.0, 500.0],
        [-1e3, -1e3, -1e3, -999.0],
        [1e3, 1e3 - 1, 1e3 - 2, -1e3],
    ])
    p = softmax(x, axis=1)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)
    # Probability mass concentrates on the largest input in each row.
    assert np.array_equal(np.argmax(p, axis=1), np.argmax(x, axis=1))


def test_encode_states_normalized():
    rng = np.random.default_rng(1)
    d = 12
    X = rng.normal(size=(4, d))
    phase_weights = np.zeros(d)
    psi = encode_states(X, phase_weights)
    norms = np.sum(np.abs(psi) ** 2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_default_bank_is_noncommuting():
    d = 12
    groups = default_feature_groups(d)
    bank = QuantumObservableBank(d, groups, seed=1)
    assert bank.commutator_norm() > 1e-6


def test_commuting_only_bank_actually_commutes():
    d = 12
    groups = default_feature_groups(d)
    bank = QuantumObservableBank(d, groups, seed=1, commuting_only=True, real_only_coeff=True)
    assert bank.commutator_norm() < 1e-9


def test_classical_limit_reduction_exact():
    """Proposition 2: with commuting observables, real coefficients, and
    covariance dropped, U_total^2 must equal sum_k alpha_k^2 Var_k exactly."""
    rng = np.random.default_rng(0)
    d = 12
    groups = default_feature_groups(d)
    bank = QuantumObservableBank(d, groups, seed=1, commuting_only=True, real_only_coeff=True)

    x = rng.normal(size=(1, d))
    x = (x - x.mean()) / (x.std() + 1e-8)
    phase_weights = np.zeros(d)
    psi = encode_states(x, phase_weights)[0]

    u_total = bank.total_uncertainty(psi, use_covariance=False)
    manual = sum(
        (bank.alpha[k].real ** 2) * bank.variance(psi, k) for k in bank.names
    )
    manual = np.sqrt(max(0.0, manual))
    assert abs(u_total - manual) < 1e-9


def test_variance_is_nonnegative():
    d = 10
    groups = default_feature_groups(d)
    bank = QuantumObservableBank(d, groups, seed=2)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, d))
    phase_weights = feature_phase_weights(X)
    psi = encode_states(X, phase_weights)
    for i in range(psi.shape[0]):
        for name in bank.names:
            assert bank.variance(psi[i], name) >= 0.0


def test_batch_scores_shape_and_finite():
    d = 10
    groups = default_feature_groups(d)
    bank = QuantumObservableBank(d, groups, seed=4)
    rng = np.random.default_rng(5)
    X_labeled = rng.normal(size=(15, d))
    X_pool = rng.normal(size=(25, d))
    scores = bank.batch_scores(X_labeled, X_pool)
    assert scores.shape == (25,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


@pytest.mark.parametrize("real_only", [True, False])
@pytest.mark.parametrize("commuting_only", [True, False])
def test_bank_construction_variants(real_only, commuting_only):
    """All ablation-relevant construction flags should produce a usable,
    finite-scoring bank without raising."""
    d = 10
    groups = default_feature_groups(d)
    bank = QuantumObservableBank(
        d, groups, seed=6, commuting_only=commuting_only, real_only_coeff=real_only
    )
    rng = np.random.default_rng(7)
    X_labeled = rng.normal(size=(10, d))
    X_pool = rng.normal(size=(5, d))
    scores = bank.batch_scores(X_labeled, X_pool)
    assert np.all(np.isfinite(scores))
