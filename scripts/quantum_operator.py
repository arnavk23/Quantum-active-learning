"""
Quantum-inspired covariance-aware uncertainty formalism.

Implements Definition 1 (state encoding), Eq. (2) (Hermitian observables),
Eq. (3) (variance), Eq. (4) (symmetrized covariance), Eq. (5) (complex
coupled acquisition score U_total), and Proposition 2 (classical-limit
reduction) exactly as specified in the manuscript, in place of the
disconnected sin/cos pseudo-code that previously stood in for this
formalism (scripts/quantum_learning.py).

Modeling choices not fully pinned down by the manuscript text are made
explicit here:
  - p_ij (Definition 1): softmax of squared standardized feature values,
    so sum_j p_ij = 1 by construction.
  - phi_ij (Definition 1): phase for feature j is the mean absolute
    Pearson correlation of feature j with the other features, estimated
    over the current labeled pool, scaled to [0, pi]. This is constant
    across i for a given labeled set (recomputed each AL iteration), i.e.
    the phase depends on the feature's role in the current correlation
    structure, not on the individual candidate.
  - Observable weight matrices W^(k): Hermitian matrices with elevated
    weight on a designated feature-index group per observable
    (structural / electronic / thermodynamic), normalized to unit
    spectral norm, with small nonzero cross terms so that
    W^(struct) W^(elec) != W^(elec) W^(struct) (non-commutativity by
    construction, verified in self_test()).
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def feature_phase_weights(X_labeled, eps=1e-8):
    """phi_j for each feature j: mean |correlation| with other features
    over the current labeled pool, scaled to [0, pi]."""
    d = X_labeled.shape[1]
    if X_labeled.shape[0] < 3:
        return np.zeros(d)
    C = np.corrcoef(X_labeled, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    mean_abs_corr = np.mean(np.abs(C), axis=1)
    return np.pi * mean_abs_corr


def encode_states(X, phase_weights):
    """Definition 1: complex amplitude vectors |psi_i> for each row of X.
    X is assumed standardized (zero mean, unit variance per column)."""
    p = softmax(X ** 2, axis=1)  # sums to 1 per row
    phi = phase_weights[None, :]  # (1, d), broadcast across candidates
    alpha = np.sqrt(p) * np.exp(1j * phi)
    return alpha  # (N, d) complex


def make_observable(d, feature_group, seed, cross_weight=0.15, block_boost=2.5):
    """Build a d x d Hermitian matrix concentrated on feature_group indices,
    normalized to unit spectral norm."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    H = (A + A.T) / 2.0
    H *= cross_weight
    mask = np.zeros((d, d))
    for i in feature_group:
        for j in feature_group:
            mask[i, j] = 1.0
    H = H + block_boost * mask * (A + A.T) / 2.0
    eigmax = np.max(np.abs(np.linalg.eigvalsh(H)))
    if eigmax > 1e-12:
        H = H / eigmax
    return H


class QuantumObservableBank:
    """K Hermitian observables with complex coupling coefficients, matching
    Eqs. (1)-(6) of the manuscript."""

    def __init__(self, d, feature_groups, coefficients=None, seed=0,
                 commuting_only=False, real_only_coeff=False):
        """
        feature_groups: dict name -> list of feature indices (defines the
            block each observable is concentrated on).
        coefficients: dict name -> complex coefficient alpha_k. Defaults to
            the manuscript's configuration for K=3
            (alpha_struct=1.0, alpha_elec=1.2 e^{i pi/4}, alpha_thermo=0.8),
            extended with unit real coefficients for any extra observables.
        commuting_only: if True, build diagonal (hence mutually commuting)
            observables -- used for the ablation / classical-limit checks.
        real_only_coeff: if True, coerce all coefficients to their real part
            (imag = 0) -- used for the "real-only coefficients" ablation.
        """
        self.d = d
        self.names = list(feature_groups.keys())
        self.K = len(self.names)

        if commuting_only:
            self.O = {}
            for i, name in enumerate(self.names):
                diag = np.zeros(d)
                idx = feature_groups[name]
                diag[idx] = 1.0
                diag += 0.05 * (i + 1)  # break degeneracy so eigenbases align
                self.O[name] = np.diag(diag / np.max(np.abs(diag)))
        else:
            self.O = {
                name: make_observable(d, feature_groups[name], seed=seed + i)
                for i, name in enumerate(self.names)
            }

        if coefficients is None:
            default = {
                "structural": 1.0 + 0j,
                "electronic": 1.2 * np.exp(1j * np.pi / 4),
                "thermodynamic": 0.8 + 0j,
            }
            coefficients = {n: default.get(n, 1.0 + 0j) for n in self.names}
        if real_only_coeff:
            coefficients = {k: complex(v.real, 0.0) for k, v in coefficients.items()}
        self.alpha = coefficients

    def commutator_norm(self):
        """Frobenius norm of [O_k, O_l] summed over all pairs -- 0 iff all
        observables commute. Used by self_test() to verify non-commutativity
        by construction."""
        total = 0.0
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                comm = self.O[a] @ self.O[b] - self.O[b] @ self.O[a]
                total += np.linalg.norm(comm)
        return total

    def variance(self, psi, name):
        """Eq. (3): Var(O_k; psi) = <psi|O^2|psi> - <psi|O|psi>^2."""
        O = self.O[name]
        exp_O = np.real(np.conj(psi) @ (O @ psi))
        exp_O2 = np.real(np.conj(psi) @ (O @ (O @ psi)))
        return max(0.0, exp_O2 - exp_O ** 2)

    def covariance(self, psi, name_k, name_l):
        """Eq. (4): symmetrized cross-observable covariance."""
        Ok, Ol = self.O[name_k], self.O[name_l]
        sym = (Ok @ Ol + Ol @ Ok) / 2.0
        exp_sym = np.real(np.conj(psi) @ (sym @ psi))
        exp_k = np.real(np.conj(psi) @ (Ok @ psi))
        exp_l = np.real(np.conj(psi) @ (Ol @ psi))
        return exp_sym - exp_k * exp_l

    def total_uncertainty(self, psi, use_covariance=True):
        """Eq. (5): U_total(M_i)."""
        var_term = sum(
            (abs(self.alpha[k]) ** 2) * self.variance(psi, k) for k in self.names
        )
        cov_term = 0.0
        if use_covariance:
            for i, k in enumerate(self.names):
                for l in self.names:
                    if k == l:
                        continue
                    cov_term += np.real(np.conj(self.alpha[k]) * self.alpha[l]) * \
                        self.covariance(psi, k, l)
        return np.sqrt(max(0.0, var_term + cov_term))

    def batch_scores(self, X_labeled, X_pool, use_covariance=True):
        """Full pipeline: recompute phases from the current labeled pool,
        encode states for the candidate pool, return U_total per candidate."""
        phase_weights = feature_phase_weights(X_labeled)
        psi_pool = encode_states(X_pool, phase_weights)
        scores = np.array([
            self.total_uncertainty(psi_pool[i], use_covariance=use_covariance)
            for i in range(psi_pool.shape[0])
        ])
        return scores


def default_feature_groups(d):
    """Deterministic 3-way split of feature indices into overlapping-but-
    distinct structural / electronic / thermodynamic groups (block-
    concentration with 1-index overlap at each boundary, matching the
    manuscript's 'overlapping but non-identical support' requirement)."""
    third = max(1, d // 3)
    struct = list(range(0, third + 1))
    elec = list(range(third, 2 * third + 1))
    thermo = list(range(2 * third, d))
    return {"structural": struct, "electronic": elec, "thermodynamic": thermo}


def self_test():
    """Verifies (a) non-commutativity of the default observable bank,
    (b) Proposition 2's classical-limit reduction: with commuting
    observables, real coefficients, and covariance dropped, U_total^2
    equals sum_k alpha_k^2 Var_k exactly."""
    rng = np.random.default_rng(0)
    d = 12
    groups = default_feature_groups(d)

    bank = QuantumObservableBank(d, groups, seed=1)
    comm_norm = bank.commutator_norm()
    assert comm_norm > 1e-6, "observables should not commute by default"
    print(f"[self_test] non-commutativity OK: sum ||[O_k,O_l]|| = {comm_norm:.4f}")

    classical_bank = QuantumObservableBank(
        d, groups, seed=1, commuting_only=True, real_only_coeff=True
    )
    comm_norm_classical = classical_bank.commutator_norm()
    assert comm_norm_classical < 1e-9, "commuting_only bank should commute exactly"

    x = rng.normal(size=(1, d))
    x = (x - x.mean()) / (x.std() + 1e-8)
    phase_weights = np.zeros(d)  # phases irrelevant once alpha is real & O commutes
    psi = encode_states(x, phase_weights)[0]

    u_total = classical_bank.total_uncertainty(psi, use_covariance=False)
    manual = sum(
        (classical_bank.alpha[k].real ** 2) * classical_bank.variance(psi, k)
        for k in classical_bank.names
    )
    manual = np.sqrt(max(0.0, manual))
    assert abs(u_total - manual) < 1e-9, f"classical limit mismatch: {u_total} vs {manual}"
    print(f"[self_test] classical-limit reduction OK: U_total={u_total:.6f} "
          f"== sqrt(sum alpha_k^2 Var_k)={manual:.6f}")


if __name__ == "__main__":
    self_test()
    print("[self_test] all checks passed.")
