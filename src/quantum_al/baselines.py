"""
Standalone extraction of the 9 legitimate baseline active-learning
implementations from scripts/benchmark.py.

benchmark.py cannot be imported directly in this environment because it
unconditionally imports quantum_learning.py at module load time, which in
turn imports qiskit (not installed, and out of scope per task
constraints). The active-learning baseline logic itself has nothing to do
with that import, so this module is a verbatim copy of the 9
`implement_*` factory methods and their nested selector classes from
scripts/benchmark.py (StateOfTheArtBenchmark), wrapped in a plain
BaselineFactory class with no qiskit dependency. No baseline logic was
rewritten -- this is a copy/relocation for reuse, not a reimplementation.

Each factory method returns an object with
    .select_next_experiments(X_candidates, X_train, y_train, n_select=10)
      -> (selected_idx, scores, info_dict)
matching the interface used by the quantum method adapter in
run_real_benchmark.py.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor


class BaselineFactory:
    """Verbatim copy of the 9 baseline `implement_*` methods from
    scripts.benchmark.StateOfTheArtBenchmark. See module docstring."""

    def implement_uncertainty_sampling(self):
        """Implement Gaussian Process uncertainty sampling."""
        class UncertaintySampling:
            def __init__(self):
                self.name = "Uncertainty Sampling"
                self.gp = None

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                try:
                    # Use Gaussian Process for uncertainty estimation
                    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
                    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
                    self.gp.fit(X_train, y_train)

                    # Predict with uncertainty
                    mean_pred, std_pred = self.gp.predict(X_candidates, return_std=True)

                    # Select highest uncertainty samples
                    uncertainty_scores = std_pred
                    selected_idx = np.argsort(uncertainty_scores)[-n_select:]

                    return selected_idx, uncertainty_scores, {'gp_uncertainty': std_pred}

                except Exception as e:
                    print(f"GP uncertainty sampling failed: {e}")
                    # Fallback to random selection
                    return np.random.choice(len(X_candidates), n_select, replace=False), None, {}

        return UncertaintySampling()

    def implement_query_by_committee(self):
        """Implement Query by Committee active learning."""
        class QueryByCommittee:
            def __init__(self):
                self.name = "Query by Committee"
                self.committee = []

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                try:
                    # Create committee of diverse models
                    i = 0
                    self.committee = [
                        RandomForestRegressor(n_estimators=50, max_depth=5, random_state=i),
                        RandomForestRegressor(n_estimators=50, max_depth=10, random_state=i+1),
                        RandomForestRegressor(n_estimators=50, max_depth=15, random_state=i+2),
                        MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=i+3),
                        MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, random_state=i+4)
                    ]

                    # Train committee
                    for model in self.committee:
                        model.fit(X_train, y_train)

                    # Get predictions from all committee members
                    predictions = []
                    for model in self.committee:
                        pred = model.predict(X_candidates)
                        predictions.append(pred)

                    predictions = np.array(predictions)

                    # Calculate disagreement (variance across committee)
                    disagreement = np.var(predictions, axis=0)

                    # Select samples with highest disagreement
                    selected_idx = np.argsort(disagreement)[-n_select:]

                    return selected_idx, disagreement, {'committee_variance': disagreement}

                except Exception as e:
                    print(f"QBC failed: {e}")
                    return np.random.choice(len(X_candidates), n_select, replace=False), None, {}

        return QueryByCommittee()

    def implement_expected_improvement(self):
        """Implement Expected Improvement (Bayesian Optimization)."""
        class ExpectedImprovement:
            def __init__(self):
                self.name = "Expected Improvement"
                self.gp = None
                self.y_best = None

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                try:
                    from scipy.stats import norm

                    # Train GP
                    kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
                    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
                    self.gp.fit(X_train, y_train)

                    # Current best value
                    self.y_best = np.max(y_train)

                    # Predict on candidates
                    mean_pred, std_pred = self.gp.predict(X_candidates, return_std=True)

                    # Calculate Expected Improvement
                    z = (mean_pred - self.y_best) / (std_pred + 1e-9)
                    ei = (mean_pred - self.y_best) * norm.cdf(z) + std_pred * norm.pdf(z)

                    # Select highest EI samples
                    selected_idx = np.argsort(ei)[-n_select:]

                    return selected_idx, ei, {'expected_improvement': ei}

                except Exception as e:
                    print(f"EI failed: {e}")
                    return np.random.choice(len(X_candidates), n_select, replace=False), None, {}

        return ExpectedImprovement()

    def implement_maximum_entropy(self):
        """Implement Maximum Entropy sampling."""
        class MaximumEntropy:
            def __init__(self):
                self.name = "Maximum Entropy"
                self.model = None

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                try:
                    # Use Random Forest for entropy estimation
                    self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                    self.model.fit(X_train, y_train)

                    # Get predictions from all trees
                    tree_predictions = []
                    for tree in self.model.estimators_:
                        pred = tree.predict(X_candidates)
                        tree_predictions.append(pred)

                    tree_predictions = np.array(tree_predictions)

                    # Estimate entropy using prediction variance
                    prediction_variance = np.var(tree_predictions, axis=0)
                    entropy_estimate = 0.5 * np.log(2 * np.pi * np.e * prediction_variance + 1e-9)

                    # Select highest entropy samples
                    selected_idx = np.argsort(entropy_estimate)[-n_select:]

                    return selected_idx, entropy_estimate, {'entropy': entropy_estimate}

                except Exception as e:
                    print(f"Max entropy failed: {e}")
                    return np.random.choice(len(X_candidates), n_select, replace=False), None, {}

        return MaximumEntropy()

    def implement_diversity_sampling(self):
        """Implement diversity-based sampling using k-means clustering."""
        class DiversitySampling:
            def __init__(self):
                self.name = "Diversity Sampling"
                self.kmeans = None

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                try:
                    # Use k-means to find diverse samples
                    self.kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)

                    # If we have too few candidates, select all
                    if len(X_candidates) <= n_select:
                        return np.arange(len(X_candidates)), None, {}

                    # Cluster candidates
                    cluster_labels = self.kmeans.fit_predict(X_candidates)

                    # Select one sample from each cluster (closest to centroid)
                    selected_idx = []
                    for i in range(n_select):
                        cluster_mask = cluster_labels == i
                        if cluster_mask.sum() > 0:
                            cluster_points = X_candidates[cluster_mask]
                            centroid = self.kmeans.cluster_centers_[i]

                            # Find closest point to centroid
                            distances = np.linalg.norm(cluster_points - centroid, axis=1)
                            closest_in_cluster = np.argmin(distances)

                            # Get global index
                            cluster_indices = np.where(cluster_mask)[0]
                            selected_idx.append(cluster_indices[closest_in_cluster])

                    selected_idx = np.array(selected_idx)
                    diversity_scores = np.zeros(len(X_candidates))
                    diversity_scores[selected_idx] = 1.0

                    return selected_idx, diversity_scores, {'diversity_selected': True}

                except Exception as e:
                    print(f"Diversity sampling failed: {e}")
                    return np.random.choice(len(X_candidates), n_select, replace=False), None, {}

        return DiversitySampling()

    def implement_badge(self):
        """Implement BADGE (Batch Active learning by Diverse Gradient Embeddings)."""
        class BADGE:
            def __init__(self):
                self.name = "BADGE"
                self.model = None

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                try:
                    # Use Random Forest as surrogate for gradient embeddings
                    self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                    self.model.fit(X_train, y_train)

                    # Use feature importance as gradient surrogate
                    feature_importance = self.model.feature_importances_

                    # Compute embeddings (feature importance weighted features)
                    embeddings = X_candidates * feature_importance

                    # Add uncertainty information
                    predictions = []
                    for tree in self.model.estimators_:
                        pred = tree.predict(X_candidates)
                        predictions.append(pred)

                    uncertainty = np.var(predictions, axis=0)

                    # Combine embeddings with uncertainty
                    enhanced_embeddings = np.column_stack([embeddings, uncertainty.reshape(-1, 1)])

                    # Use k-means++ initialization for diverse selection
                    kmeans = KMeans(n_clusters=n_select, init='k-means++', random_state=42, n_init=10)

                    if len(enhanced_embeddings) <= n_select:
                        return np.arange(len(X_candidates)), None, {}

                    cluster_labels = kmeans.fit_predict(enhanced_embeddings)

                    # Select samples closest to cluster centers
                    selected_idx = []
                    for i in range(n_select):
                        cluster_mask = cluster_labels == i
                        if cluster_mask.sum() > 0:
                            cluster_points = enhanced_embeddings[cluster_mask]
                            centroid = kmeans.cluster_centers_[i]

                            distances = np.linalg.norm(cluster_points - centroid, axis=1)
                            closest_in_cluster = np.argmin(distances)

                            cluster_indices = np.where(cluster_mask)[0]
                            selected_idx.append(cluster_indices[closest_in_cluster])

                    selected_idx = np.array(selected_idx)
                    badge_scores = np.zeros(len(X_candidates))
                    badge_scores[selected_idx] = uncertainty[selected_idx]

                    return selected_idx, badge_scores, {'badge_embeddings': True}

                except Exception as e:
                    print(f"BADGE failed: {e}")
                    return np.random.choice(len(X_candidates), n_select, replace=False), None, {}

        return BADGE()

    def implement_coreset(self):
        """Implement CoreSet selection."""
        class CoreSet:
            def __init__(self):
                self.name = "CoreSet"

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                try:
                    # Use greedy k-center algorithm
                    selected_idx = []
                    remaining_candidates = list(range(len(X_candidates)))

                    # Start with random point
                    first_idx = np.random.randint(len(remaining_candidates))
                    selected_idx.append(remaining_candidates[first_idx])
                    remaining_candidates.remove(remaining_candidates[first_idx])

                    for _ in range(n_select - 1):
                        if not remaining_candidates:
                            break

                        max_min_distance = -1
                        best_candidate = None

                        for candidate_idx in remaining_candidates:
                            # Calculate minimum distance to already selected points
                            min_distance = float('inf')

                            # Distance to training points
                            for train_idx in range(len(X_train)):
                                dist = np.linalg.norm(X_candidates[candidate_idx] - X_train[train_idx])
                                min_distance = min(min_distance, dist)

                            # Distance to selected candidates
                            for selected_candidate_idx in selected_idx:
                                dist = np.linalg.norm(X_candidates[candidate_idx] - X_candidates[selected_candidate_idx])
                                min_distance = min(min_distance, dist)

                            # Select candidate with maximum minimum distance
                            if min_distance > max_min_distance:
                                max_min_distance = min_distance
                                best_candidate = candidate_idx

                        if best_candidate is not None:
                            selected_idx.append(best_candidate)
                            remaining_candidates.remove(best_candidate)

                    selected_idx = np.array(selected_idx)
                    coreset_scores = np.zeros(len(X_candidates))
                    coreset_scores[selected_idx] = 1.0

                    return selected_idx, coreset_scores, {'coreset_selected': True}

                except Exception as e:
                    print(f"CoreSet failed: {e}")
                    return np.random.choice(len(X_candidates), n_select, replace=False), None, {}

        return CoreSet()

    def implement_rf_uncertainty(self):
        """Implement Random Forest uncertainty sampling."""
        class RandomForestUncertainty:
            def __init__(self):
                self.name = "RF Uncertainty"
                self.rf = None

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                try:
                    # Train Random Forest
                    self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    self.rf.fit(X_train, y_train)

                    # Get predictions from all trees
                    tree_predictions = []
                    for tree in self.rf.estimators_:
                        pred = tree.predict(X_candidates)
                        tree_predictions.append(pred)

                    tree_predictions = np.array(tree_predictions)

                    # Calculate variance across trees as uncertainty
                    uncertainty = np.var(tree_predictions, axis=0)

                    # Select highest uncertainty samples
                    selected_idx = np.argsort(uncertainty)[-n_select:]

                    return selected_idx, uncertainty, {'rf_uncertainty': uncertainty}

                except Exception as e:
                    print(f"RF uncertainty failed: {e}")
                    return np.random.choice(len(X_candidates), n_select, replace=False), None, {}

        return RandomForestUncertainty()

    def implement_random_sampling(self):
        """Implement random sampling baseline."""
        class RandomSampling:
            def __init__(self):
                self.name = "Random Sampling"

            def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
                n_available = len(X_candidates)
                n_select = min(n_select, n_available)
                selected_idx = np.random.choice(n_available, n_select, replace=False)
                scores = np.random.random(n_available)
                return selected_idx, scores, {'random_selection': True}

        return RandomSampling()


BASELINE_METHODS = [
    ("Uncertainty Sampling", "implement_uncertainty_sampling"),
    ("Query by Committee", "implement_query_by_committee"),
    ("Expected Improvement", "implement_expected_improvement"),
    ("Maximum Entropy", "implement_maximum_entropy"),
    ("Diversity Sampling", "implement_diversity_sampling"),
    ("BADGE", "implement_badge"),
    ("CoreSet", "implement_coreset"),
    ("RF Uncertainty", "implement_rf_uncertainty"),
    ("Random Sampling", "implement_random_sampling"),
]


def get_all_baselines():
    """Return dict name -> fresh selector instance for all 9 baselines."""
    factory = BaselineFactory()
    return {name: getattr(factory, method)() for name, method in BASELINE_METHODS}
