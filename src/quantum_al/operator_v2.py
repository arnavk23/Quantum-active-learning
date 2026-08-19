"""
Principled improvement attempt over quantum_operator.py, built AFTER seeing
that the faithful implementation of the paper's Eq. (1)-(6) does not beat
baselines on real data (results/primary_benchmark_summary.json).

Diagnosis: quantum_operator.py's U_total(M_i) depends only on the
candidate's raw standardized feature vector and the labeled pool's
correlation structure -- never on which features the current downstream
model actually finds predictive. Every competitive baseline (Uncertainty
Sampling, QBC, EI, Maximum Entropy, RF Uncertainty) instead measures
uncertainty of/through a model fit to (X_train, y_train). That is the most
likely reason for the gap, not the covariance/complex-coefficient
machinery (the ablation already showed those contribute ~noise-level
deltas).

This module keeps quantum_operator.py's core formalism (Hermitian
observables, variance, covariance, complex coupling) completely intact and
unmodified, and adds exactly two independently-testable changes on top,
so their effect can be isolated:

  1. Domain-informed feature groups (`domain_feature_groups`): the original
     default_feature_groups() splits the 21-d feature vector into arbitrary
     contiguous index thirds, which mixes semantically unrelated columns
     (e.g. electronegativity ends up partly in the "structural" group).
     This groups by actual feature semantics instead.

  2. Predictor-informed rescaling (`ImportanceWeightedQuantumSelector`):
     fits a small RandomForest on the current (X_train, y_train) each
     iteration (same information every baseline already uses) and rescales
     each feature column by sqrt(importance) before state encoding, so the
     Hilbert-space embedding -- and hence every variance/covariance term
     computed on top of it -- is weighted toward directions the current
     model finds predictive, rather than treating all 21 standardized
     features as equally salient.

Both changes are reported honestly regardless of outcome; see
results/improvement_attempt.json / results/SUMMARY.md for what actually
happened when this was run.
"""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from quantum_al.operator import (
    QuantumObservableBank, encode_states, feature_phase_weights, softmax,
)
from quantum_al.data_utils import FEATURE_COLUMNS

DOMAIN_GROUPS_BY_NAME = {
    # geometry / coordination
    "structural": ["density", "volume_per_atom", "nsites", "space_group_number",
                   "atomic_radius_mean", "atomic_radius_std", "atomic_radius_range"],
    # electronegativity / valence character (group number in the periodic
    # table correlates with valence electron count)
    "electronic": ["X_mean", "X_std", "X_range",
                   "group_mean", "group_std", "group_range"],
    # compositional complexity / thermodynamic stability
    "thermodynamic": ["nelements", "energy_above_hull",
                       "atomic_mass_mean", "atomic_mass_std", "atomic_mass_range",
                       "row_mean", "row_std", "row_range"],
}


def domain_feature_groups(feature_columns=FEATURE_COLUMNS):
    idx = {name: i for i, name in enumerate(feature_columns)}
    return {
        group: [idx[c] for c in cols if c in idx]
        for group, cols in DOMAIN_GROUPS_BY_NAME.items()
    }


class ImportanceWeightedQuantumSelector:
    """Same acquisition rule as QuantumSelector (rank by U_total), but the
    feature vector is rescaled by sqrt(RandomForest feature_importances_)
    fit on the current labeled set before state encoding. Falls back to
    unweighted (all-ones) scaling if the RF fit fails for any reason."""

    def __init__(self, d, feature_groups=None, seed=0, use_covariance=True,
                 n_estimators=50, name="Quantum-ImportanceWeighted"):
        self.name = name
        groups = feature_groups if feature_groups is not None else domain_feature_groups()
        self.bank = QuantumObservableBank(d, groups, seed=seed)
        self.use_covariance = use_covariance
        self.n_estimators = n_estimators

    def _importance_weights(self, X_train, y_train, d):
        """Falls back to uniform weighting only for the specific,
        anticipated case of too few labeled samples for RF fitting
        (ValueError from scikit-learn) -- not a blanket except, so
        unexpected bugs (e.g. shape mismatches from a caller error) still
        raise instead of being silently masked."""
        try:
            rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
            rf.fit(X_train, y_train)
        except ValueError as e:
            warnings.warn(
                f"RandomForest fit failed ({e}); falling back to uniform "
                "importance weights for this iteration."
            )
            return np.ones(d)
        imp = rf.feature_importances_
        imp = np.clip(imp, 1e-6, None)
        return np.sqrt(imp / imp.mean())

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        d = X_train.shape[1]
        w = self._importance_weights(X_train, y_train, d)
        X_train_w = X_train * w[None, :]
        X_cand_w = X_candidates * w[None, :]

        phase_weights = feature_phase_weights(X_train_w)
        psi_pool = encode_states(X_cand_w, phase_weights)
        scores = np.array([
            self.bank.total_uncertainty(psi_pool[i], use_covariance=self.use_covariance)
            for i in range(psi_pool.shape[0])
        ])
        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {"quantum_scores": scores, "importance_weights": w.tolist()}
