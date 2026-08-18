"""
Real quantum circuit realization of the state encoding + observable
formalism in quantum_operator.py (Eqs. 1-6), as opposed to a classical
linear-algebra simulation of the same math.

Everything in quantum_operator.py and the primary/ablation/v3 benchmarks
is quantum-INSPIRED: the "quantum state" |psi_i> is a numpy complex
vector, and "observables" are numpy matrices manipulated via ordinary
linear algebra -- no circuit, no qubit, no measurement ever appears. This
module builds the thing the paper's own "Quantum hardware realization"
future-work paragraph gestures at but never does: an actual circuit that
(a) prepares |psi_i> via amplitude encoding on ceil(log2(d)) qubits, and
(b) estimates <O>, <O^2>, and the covariance cross-terms via Pauli
decomposition + measurement, exactly as a real NISQ device would have to.

Two honest, separate things are reported, not conflated:
  1. Correctness: in the noiseless, infinite-shot (exact statevector)
     limit, does the circuit reproduce the classical closed-form
     Var/Cov/U_total exactly? (It must, or the "formalism" and the
     "circuit realization" are not the same object.)
  2. NISQ feasibility: qubit count, transpiled gate count/depth, number
     of distinct Pauli measurement groups, and shot-count required for
     the resulting acquisition RANKING (not just the raw numbers) to
     match the exact ranking, with and without a simple depolarizing
     noise model. This is a resource/feasibility characterization, not a
     performance claim -- it may come out favorable or unfavorable, and
     is reported either way.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator

try:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    _HAVE_AER = True
except ImportError:
    _HAVE_AER = False


def n_qubits_for_dim(d):
    return int(np.ceil(np.log2(max(d, 2))))


def pad_amplitudes(alpha, n_qubits):
    """Zero-pad a length-d complex amplitude vector to length 2^n_qubits
    and renormalize (padding with exact zeros doesn't change the norm,
    but we renormalize defensively against floating-point drift)."""
    dim = 2 ** n_qubits
    padded = np.zeros(dim, dtype=complex)
    padded[: len(alpha)] = alpha
    norm = np.linalg.norm(padded)
    if norm > 1e-12:
        padded = padded / norm
    return padded


def pad_observable(O, n_qubits):
    """Zero-pad a d x d Hermitian matrix to 2^n_qubits x 2^n_qubits
    (the padded basis states carry zero amplitude, so they never
    contribute to any expectation value; padding is purely to give the
    operator a well-defined action on the full qubit Hilbert space)."""
    dim = 2 ** n_qubits
    d = O.shape[0]
    padded = np.zeros((dim, dim), dtype=complex)
    padded[:d, :d] = O
    return padded


def amplitude_encoding_circuit(alpha, n_qubits=None):
    """Real state-preparation circuit for |psi> = sum_j alpha_j |j>."""
    if n_qubits is None:
        n_qubits = n_qubits_for_dim(len(alpha))
    padded = pad_amplitudes(alpha, n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.append(StatePreparation(padded), range(n_qubits))
    return qc, padded


def pauli_decompose(O_padded):
    """SparsePauliOp decomposition of a Hermitian matrix. Returns the
    operator plus the number of nonzero Pauli terms (a direct measure of
    NISQ measurement overhead: each term generally needs a distinct
    measurement basis / circuit).

    NOTE: SparsePauliOp.from_operator truncates small coefficients by
    default (atol/rtol=None resolves to a nonzero default), which we
    found introduces ~1e-5 reconstruction error for operators like O^2
    with a wider coefficient magnitude range than O itself -- silently
    breaking exact agreement with the classical closed-form. We pass
    atol=0, rtol=0 explicitly to get the true, lossless decomposition,
    then drop only genuinely-zero (float roundoff, <1e-12) terms
    ourselves.
    """
    op = SparsePauliOp.from_operator(O_padded, atol=0, rtol=0)
    mask = np.abs(op.coeffs) > 1e-12
    op = SparsePauliOp(op.paulis[mask], op.coeffs[mask])
    return op


def exact_circuit_expectation(qc, pauli_op):
    """Noiseless, infinite-shot expectation value via exact statevector
    simulation of the actual circuit (StatePreparation + measurement),
    using Qiskit's reference Estimator -- this is what the circuit
    computes in the ideal limit, not a shortcut."""
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, pauli_op)])
    result = job.result()[0]
    return float(np.real(result.data.evs))


def shot_based_expectation(qc, pauli_op, shots, noise_model=None, seed=0):
    """Finite-shot expectation value estimate via AerEstimatorV2. `shots`
    is converted to an equivalent target precision (1/sqrt(shots)),
    matching how a real device's measurement statistics would scale.
    Aer's simulator backend does not accept the high-level
    StatePreparation instruction directly, so the circuit is decomposed
    to elementary gates first (this is exactly what a real backend's
    transpiler would do before execution)."""
    if not _HAVE_AER:
        raise RuntimeError("qiskit-aer is required for shot-based simulation")
    backend_options = {"seed_simulator": seed}
    if noise_model is not None:
        backend_options["noise_model"] = noise_model
    estimator = AerEstimatorV2(options={"backend_options": backend_options})
    qc_decomposed = qc.decompose(reps=5)
    job = estimator.run([(qc_decomposed, pauli_op)], precision=1.0 / np.sqrt(shots))
    result = job.result()[0]
    return float(np.real(result.data.evs))


def circuit_variance(alpha, O, n_qubits=None, exact=True, shots=None, noise_model=None, seed=0):
    """Var(O; psi) = <O^2> - <O>^2, computed via real circuits (state
    prep + Pauli-decomposed measurement of O and O^2 separately)."""
    if n_qubits is None:
        n_qubits = n_qubits_for_dim(len(alpha))
    qc, padded_alpha = amplitude_encoding_circuit(alpha, n_qubits)
    O_pad = pad_observable(O, n_qubits)
    O2_pad = O_pad @ O_pad

    op_O = pauli_decompose(O_pad)
    op_O2 = pauli_decompose(O2_pad)

    if exact:
        exp_O = exact_circuit_expectation(qc, op_O)
        exp_O2 = exact_circuit_expectation(qc, op_O2)
    else:
        exp_O = shot_based_expectation(qc, op_O, shots, noise_model, seed)
        exp_O2 = shot_based_expectation(qc, op_O2, shots, noise_model, seed + 1)

    var = exp_O2 - exp_O ** 2
    return max(0.0, var), len(op_O.paulis), len(op_O2.paulis)


def circuit_covariance(alpha, Ok, Ol, n_qubits=None, exact=True, shots=None, noise_model=None, seed=0):
    if n_qubits is None:
        n_qubits = n_qubits_for_dim(len(alpha))
    qc, padded_alpha = amplitude_encoding_circuit(alpha, n_qubits)
    Ok_pad = pad_observable(Ok, n_qubits)
    Ol_pad = pad_observable(Ol, n_qubits)
    sym = (Ok_pad @ Ol_pad + Ol_pad @ Ok_pad) / 2.0

    op_k = pauli_decompose(Ok_pad)
    op_l = pauli_decompose(Ol_pad)
    op_sym = pauli_decompose(sym)

    if exact:
        exp_sym = exact_circuit_expectation(qc, op_sym)
        exp_k = exact_circuit_expectation(qc, op_k)
        exp_l = exact_circuit_expectation(qc, op_l)
    else:
        exp_sym = shot_based_expectation(qc, op_sym, shots, noise_model, seed)
        exp_k = shot_based_expectation(qc, op_k, shots, noise_model, seed + 1)
        exp_l = shot_based_expectation(qc, op_l, shots, noise_model, seed + 2)

    cov = exp_sym - exp_k * exp_l
    return cov, len(op_sym.paulis)


def transpiled_resource_report(alpha, basis_gates=("cx", "rz", "sx", "x")):
    """Qubit count, transpiled gate count, and circuit depth for the
    amplitude-encoding state-preparation circuit alone (the Pauli
    measurement circuits add a small constant number of single-qubit
    basis-change gates per term on top of this)."""
    from qiskit import transpile
    n_qubits = n_qubits_for_dim(len(alpha))
    qc, _ = amplitude_encoding_circuit(alpha, n_qubits)
    tqc = transpile(qc, basis_gates=list(basis_gates), optimization_level=1)
    gate_counts = dict(tqc.count_ops())
    return {
        "n_qubits": n_qubits,
        "depth": tqc.depth(),
        "gate_counts": gate_counts,
        "total_gates": sum(gate_counts.values()),
        "cx_count": gate_counts.get("cx", 0),
    }


def make_depolarizing_noise_model(p1=0.001, p2=0.01):
    """Simple, standard 1-/2-qubit depolarizing noise model at
    representative near-term hardware error rates."""
    if not _HAVE_AER:
        return None
    noise_model = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)
    noise_model.add_all_qubit_quantum_error(err1, ["sx", "x", "rz"])
    noise_model.add_all_qubit_quantum_error(err2, ["cx"])
    return noise_model


def self_test():
    """Verifies the circuit-computed Var/Cov match the classical
    closed-form (operator.py) exactly in the noiseless limit."""
    from quantum_al.operator import make_observable, default_feature_groups, encode_states

    rng = np.random.default_rng(0)
    d = 21
    x = rng.normal(size=(1, d))
    x = (x - x.mean()) / (x.std() + 1e-8)
    phase_weights = np.zeros(d)
    alpha = encode_states(x, phase_weights)[0]

    groups = default_feature_groups(d)
    O_struct = make_observable(d, groups["structural"], seed=1)
    O_elec = make_observable(d, groups["electronic"], seed=2)

    # Classical closed-form reference.
    exp_O = np.real(np.conj(alpha) @ (O_struct @ alpha))
    exp_O2 = np.real(np.conj(alpha) @ (O_struct @ (O_struct @ alpha)))
    classical_var = max(0.0, exp_O2 - exp_O ** 2)

    circuit_var, n_terms_O, n_terms_O2 = circuit_variance(alpha, O_struct, exact=True)
    print(f"[self_test] classical Var = {classical_var:.8f}")
    print(f"[self_test] circuit   Var = {circuit_var:.8f}  "
          f"({n_terms_O} Pauli terms for O, {n_terms_O2} for O^2)")
    assert abs(classical_var - circuit_var) < 1e-6, "circuit/classical variance mismatch"

    sym = (O_struct @ O_elec + O_elec @ O_struct) / 2.0
    exp_sym = np.real(np.conj(alpha) @ (sym @ alpha))
    exp_k = np.real(np.conj(alpha) @ (O_struct @ alpha))
    exp_l = np.real(np.conj(alpha) @ (O_elec @ alpha))
    classical_cov = exp_sym - exp_k * exp_l

    circuit_cov, n_terms_cov = circuit_covariance(alpha, O_struct, O_elec, exact=True)
    print(f"[self_test] classical Cov = {classical_cov:.8f}")
    print(f"[self_test] circuit   Cov = {circuit_cov:.8f}  ({n_terms_cov} Pauli terms)")
    assert abs(classical_cov - circuit_cov) < 1e-6, "circuit/classical covariance mismatch"

    resources = transpiled_resource_report(alpha)
    print(f"[self_test] state-prep resources: {resources}")

    print("[self_test] circuit realization matches classical formalism exactly. All checks passed.")


if __name__ == "__main__":
    self_test()
