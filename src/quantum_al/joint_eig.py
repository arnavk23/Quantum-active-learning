"""Joint expected-information-gain (JEIG) acquisition for K correlated
real targets, derived from Bayesian experimental design rather than the
quantum-operator analogy in operator.py.

Model: a random-forest ensemble gives, for each candidate x, M per-tree
multi-output predictions f_1(x),...,f_M(x) in R^K. Their empirical
covariance Sigma_pred(x) approximates the epistemic (model) uncertainty
over the joint prediction; R is the diagonal aleatoric/residual variance
per task (estimated out-of-bag). Under a Gaussian approximation this
gives the standard ensemble-disagreement estimate of expected information
gain (the multivariate generalization of BALD):

    JEIG(x) = 1/2 * log det(Sigma_pred(x) + R)

Proposition (checked in self_test): this always satisfies

    sum_k EIG_k(x) - JEIG(x) = -1/2 * log det(Corr(x)) >= 0

where EIG_k is the per-task marginal term and Corr(x) is Sigma_pred(x)+R
rescaled to unit diagonal. The gap is the total correlation among the K
predictive uncertainties: zero iff they are uncorrelated (JEIG reduces to
scoring by summed marginal EIGs, i.e. ordinary per-task ensemble variance,
recovering the classical baseline as a special case), strictly positive
whenever tasks share epistemic uncertainty. This is a real inequality
(Hadamard's determinant inequality), not an empirical claim.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def per_tree_predictions(forest, X):
    """(n_trees, n_samples, n_outputs) predictions, one row per tree."""
    preds = np.stack([est.predict(X) for est in forest.estimators_], axis=0)
    if preds.ndim == 2:
        preds = preds[:, :, None]
    return preds


def predictive_covariance(forest, X):
    """Per-sample (n_outputs, n_outputs) covariance across trees.
    Returns array of shape (n_samples, K, K)."""
    preds = per_tree_predictions(forest, X)  # (M, n, K)
    M, n, K = preds.shape
    mean = preds.mean(axis=0, keepdims=True)  # (1, n, K)
    centered = preds - mean  # (M, n, K)
    # Sigma[i] = (1/(M-1)) * centered[:,i,:].T @ centered[:,i,:]
    Sigma = np.einsum("mik,mil->ikl", centered, centered) / max(M - 1, 1)
    return Sigma


def oob_residual_variance(forest, X, Y):
    """Diagonal (K,) out-of-bag residual variance per task, i.e. the
    aleatoric/noise floor R used in Sigma_pred + R."""
    if not forest.oob_prediction_.shape:
        raise ValueError("forest must be fit with oob_score=True")
    resid = Y - forest.oob_prediction_
    if resid.ndim == 1:
        resid = resid[:, None]
    var = resid.var(axis=0)
    return np.maximum(var, 1e-8)


def joint_eig_score(Sigma_pred, R):
    """0.5*logdet(Sigma_pred(x) + R) per candidate. Sigma_pred: (n,K,K).
    total = Sigma_pred (PSD, sample covariance) + diag(R) (R > 0) is
    always strictly positive definite, so slogdet's sign is always +1."""
    total = Sigma_pred + np.diag(R)[None, :, :]
    sign, logdet = np.linalg.slogdet(total)
    assert np.all(sign > 0), "Sigma_pred + diag(R) must be positive definite"
    return 0.5 * logdet


def marginal_sum_score(Sigma_pred, R):
    """sum_k 0.5*log(Sigma_pred_kk(x) + R_k) per candidate: the classical
    limit that ignores cross-task correlation (ordinary per-task ensemble
    variance, summed)."""
    diag = np.diagonal(Sigma_pred, axis1=1, axis2=2)  # (n, K)
    total_diag = diag + R[None, :]
    return 0.5 * np.log(np.maximum(total_diag, 1e-300)).sum(axis=1)


def total_correlation_gap(Sigma_pred, R):
    """marginal_sum_score - joint_eig_score, elementwise. Always >= 0
    (Hadamard's inequality) and equals -0.5*logdet(correlation matrix)."""
    return marginal_sum_score(Sigma_pred, R) - joint_eig_score(Sigma_pred, R)


def self_test():
    rng = np.random.default_rng(0)
    n, K = 25, 3
    # random SPD covariances per sample
    for _ in range(20):
        A = rng.normal(size=(n, K, K))
        Sigma = np.einsum("nij,nkj->nik", A, A) + 1e-3 * np.eye(K)[None]
        R = np.abs(rng.normal(size=K)) + 1e-2

        joint = joint_eig_score(Sigma, R)
        marginal = marginal_sum_score(Sigma, R)
        gap = total_correlation_gap(Sigma, R)

        assert np.all(gap >= -1e-9), "Hadamard inequality violated"

        total = Sigma + np.diag(R)[None]
        d = np.diagonal(total, axis1=1, axis2=2)
        corr = total / np.sqrt(d[:, :, None] * d[:, None, :])
        sign, logdet_corr = np.linalg.slogdet(corr)
        assert np.all(sign > 0)
        expected_gap = -0.5 * logdet_corr
        assert np.allclose(gap, expected_gap, atol=1e-8), (gap, expected_gap)

        assert np.allclose(marginal - joint, gap, atol=1e-10)

    # diagonal Sigma_pred (uncorrelated tasks) -> gap exactly zero
    Sigma_diag = np.zeros((5, K, K))
    for i in range(5):
        Sigma_diag[i] = np.diag(rng.uniform(0.1, 2.0, size=K))
    R = np.ones(K) * 0.5
    gap_diag = total_correlation_gap(Sigma_diag, R)
    assert np.allclose(gap_diag, 0.0, atol=1e-10), gap_diag

    print("joint_eig self_test passed")
    return True


class JointEIGSelector:
    """Ranks candidates by joint expected information gain across K
    real, jointly-labeled targets, using a multi-output random forest
    ensemble as the predictive model."""

    def __init__(self, n_estimators=200, seed=0, name="Joint-EIG"):
        self.n_estimators = n_estimators
        self.seed = seed
        self.name = name

    def _fit(self, X_train, Y_train):
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.seed,
            oob_score=True, n_jobs=-1,
        )
        forest.fit(X_train, Y_train)
        R = oob_residual_variance(forest, X_train, Y_train)
        return forest, R

    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        forest, R = self._fit(X_train, Y_train)
        Sigma_pred = predictive_covariance(forest, X_candidates)
        scores = joint_eig_score(Sigma_pred, R)
        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {"forest": forest, "R": R}


class MarginalSumSelector:
    """Classical-limit baseline: same ensemble, ranks by summed per-task
    ensemble variance, ignoring cross-task correlation."""

    def __init__(self, n_estimators=200, seed=0, name="Marginal-Sum"):
        self.n_estimators = n_estimators
        self.seed = seed
        self.name = name

    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.seed,
            oob_score=True, n_jobs=-1,
        )
        forest.fit(X_train, Y_train)
        R = oob_residual_variance(forest, X_train, Y_train)
        Sigma_pred = predictive_covariance(forest, X_candidates)
        scores = marginal_sum_score(Sigma_pred, R)
        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {"forest": forest, "R": R}


if __name__ == "__main__":
    self_test()
