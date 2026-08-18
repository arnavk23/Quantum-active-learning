"""
Correctness test for the real quantum circuit realization
(quantum_al.circuit). Verifies the circuit-measured variance and
covariance, computed via exact statevector simulation, match the
classical closed-form formalism (quantum_al.operator) to floating-point
precision. This is the check that confirms the circuit and the classical
formula compute the same mathematical object, not merely an analogous
one; see the "Quantum Hardware Realization" section of papers/.

Skipped automatically if qiskit / qiskit-aer are not installed (they are
an optional dependency, `pip install -e ".[circuit]"`).
"""
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from quantum_al.operator import make_observable, default_feature_groups, encode_states
from quantum_al.circuit import circuit_variance, circuit_covariance


@pytest.fixture
def sample_state():
    rng = np.random.default_rng(0)
    d = 21
    x = rng.normal(size=(1, d))
    x = (x - x.mean()) / (x.std() + 1e-8)
    phase_weights = np.zeros(d)
    alpha = encode_states(x, phase_weights)[0]
    groups = default_feature_groups(d)
    O_struct = make_observable(d, groups["structural"], seed=1)
    O_elec = make_observable(d, groups["electronic"], seed=2)
    return alpha, O_struct, O_elec


def test_circuit_variance_matches_classical(sample_state):
    alpha, O_struct, _ = sample_state
    exp_O = np.real(np.conj(alpha) @ (O_struct @ alpha))
    exp_O2 = np.real(np.conj(alpha) @ (O_struct @ (O_struct @ alpha)))
    classical_var = max(0.0, exp_O2 - exp_O ** 2)

    circuit_var, n_terms_O, n_terms_O2 = circuit_variance(alpha, O_struct, exact=True)

    assert abs(classical_var - circuit_var) < 1e-6
    assert n_terms_O > 0 and n_terms_O2 > 0


def test_circuit_covariance_matches_classical(sample_state):
    alpha, O_struct, O_elec = sample_state
    sym = (O_struct @ O_elec + O_elec @ O_struct) / 2.0
    exp_sym = np.real(np.conj(alpha) @ (sym @ alpha))
    exp_k = np.real(np.conj(alpha) @ (O_struct @ alpha))
    exp_l = np.real(np.conj(alpha) @ (O_elec @ alpha))
    classical_cov = exp_sym - exp_k * exp_l

    circuit_cov, n_terms = circuit_covariance(alpha, O_struct, O_elec, exact=True)

    assert abs(classical_cov - circuit_cov) < 1e-6
    assert n_terms > 0
