"""
Observable Sensitivity Analysis for Quantum-Enhanced Active Learning

Explores sensitivity of performance to:
1. Number of observables (2, 3, 4, 5, 6)
2. Choice of observable types (structural, electronic, thermodynamic, magnetic, optical)
3. Observable correlation structure (independent vs. correlated)
4. Observable weight initialization strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import json


class ObservableSensitivityAnalysis:
    """
    Comprehensive sensitivity analysis for quantum observables:
    - Number of observables
    - Observable types and combinations
    - Correlation structure
    - Weight initialization
    """
    
    def __init__(self):
        self.observable_types = {
            'structural': {'desc': 'Atomic positions, symmetry', 'base_weight': 0.5},
            'electronic': {'desc': 'Band structure, DOS', 'base_weight': 0.4},
            'thermodynamic': {'desc': 'Formation energy, stability', 'base_weight': 0.6},
            'magnetic': {'desc': 'Spin configuration', 'base_weight': 0.3},
            'optical': {'desc': 'Dielectric, refractive index', 'base_weight': 0.35}
        }
        
    def generate_dataset(self, n_samples=600, n_features=20, seed=42):
        """Generate synthetic materials dataset."""
        np.random.seed(seed)
        X = np.random.randn(n_samples, n_features)
        y = 2.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + np.random.randn(n_samples) * 0.3
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y
    
    def construct_observable_matrix(self, observable_type, n_features=20, seed=42):
        """
        Construct Hermitian observable matrix for a given type.
        Different types have different feature interaction patterns.
        """
        np.random.seed(seed + hash(observable_type) % 2**32)
        
        # Random Hermitian matrix
        A = np.random.randn(n_features, n_features)
        H = (A + A.T) / 2  # Make Hermitian
        
        # Type-specific scaling and structure
        base_weight = self.observable_types[observable_type]['base_weight']
        
        if observable_type == 'structural':
            # Emphasize local correlations (tridiagonal-ish)
            mask = np.abs(np.arange(n_features)[:, None] - np.arange(n_features)) <= 2
            H = H * mask
        
        elif observable_type == 'electronic':
            # Long-range correlations
            H = H * (1 + 0.3 * np.random.rand(n_features, n_features))
        
        elif observable_type == 'thermodynamic':
            # Dominant diagonal (energetic stability)
            H = H + 2.0 * np.diag(np.random.rand(n_features))
        
        elif observable_type == 'magnetic':
            # Block-diagonal (spin sectors)
            block_size = n_features // 4
            for i in range(4):
                start = i * block_size
                end = min((i+1) * block_size, n_features)
                H[start:end, start:end] *= 1.5
        
        elif observable_type == 'optical':
            # Random with medium correlations
            H = H * 0.8
        
        # Normalize
        eigenvalues = np.linalg.eigvalsh(H)
        H = H * base_weight / (np.max(np.abs(eigenvalues)) + 1e-10)
        
        return H
    
    def compute_observable_uncertainty(self, feature_vector, observable_matrix):
        """Compute variance of an observable on a feature state."""
        # Variance: <ψ|O²|ψ> - <ψ|O|ψ>²
        expectation = feature_vector @ observable_matrix @ feature_vector.T
        expectation_sq = feature_vector @ (observable_matrix @ observable_matrix) @ feature_vector.T
        variance = expectation_sq - expectation**2
        return max(0, variance)  # Ensure non-negative
    
    def aggregate_multi_observable_uncertainty(self, feature_vector, observable_matrices, 
                                              correlation_structure='correlated'):
        """
        Aggregate uncertainty across multiple observables.
        Correlation structure: 'independent' or 'correlated'
        """
        n_observables = len(observable_matrices)
        
        # Individual variances
        variances = [self.compute_observable_uncertainty(feature_vector, obs) 
                    for obs in observable_matrices]
        
        if correlation_structure == 'independent':
            # Simple sum (no covariance)
            total_uncertainty = np.sqrt(np.sum(variances))
        
        else:  # correlated
            # Include covariance terms
            covariances = []
            for i in range(n_observables):
                for j in range(i+1, n_observables):
                    # Approximate covariance via symmetrized product
                    O_i = observable_matrices[i]
                    O_j = observable_matrices[j]
                    
                    cov_ij = 0.5 * (
                        feature_vector @ (O_i @ O_j) @ feature_vector.T +
                        feature_vector @ (O_j @ O_i) @ feature_vector.T
                    ) - (feature_vector @ O_i @ feature_vector.T) * (feature_vector @ O_j @ feature_vector.T)
                    
                    covariances.append(cov_ij)
            
            total_uncertainty = np.sqrt(np.sum(variances) + np.sum(covariances))
        
        return total_uncertainty
    
    def run_with_n_observables(self, n_observables, observable_types_list, 
                               correlation_structure='correlated',
                               n_iterations=6, batch_size=15, initial_pool_size=40):
        """Run active learning with specified number and types of observables."""
        
        # Generate dataset
        X, y = self.generate_dataset(n_samples=600)
        
        # Split
        split_idx = int(0.7 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        # Construct observables
        observable_matrices = [
            self.construct_observable_matrix(obs_type, n_features=X.shape[1])
            for obs_type in observable_types_list[:n_observables]
        ]
        
        # Active learning loop
        labeled_idx = np.random.choice(len(X_train), size=initial_pool_size, replace=False)
        pool_idx = np.array([i for i in range(len(X_train)) if i not in labeled_idx])
        
        r2_scores = []
        
        for iteration in range(n_iterations):
            # Train model
            model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                n_restarts_optimizer=3,
                alpha=1e-6,
                normalize_y=True,
                random_state=42
            )
            
            model.fit(X_train[labeled_idx], y_train[labeled_idx])
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)
            
            # Query next batch
            if len(pool_idx) > batch_size:
                uncertainties = []
                
                for idx in pool_idx:
                    feature_vec = X_train[idx:idx+1].flatten()
                    
                    # Multi-observable uncertainty
                    unc = self.aggregate_multi_observable_uncertainty(
                        feature_vec, observable_matrices, correlation_structure
                    )
                    uncertainties.append(unc)
                
                uncertainties = np.array(uncertainties)
                top_indices = np.argsort(uncertainties)[-batch_size:]
                selected = pool_idx[top_indices]
                
                labeled_idx = np.concatenate([labeled_idx, selected])
                pool_idx = np.array([i for i in pool_idx if i not in selected])
        
        return r2_scores
    
    def sensitivity_to_number_of_observables(self):
        """Test sensitivity to number of observables (2 to 6)."""
        print("\n" + "="*70)
        print("SENSITIVITY TO NUMBER OF OBSERVABLES")
        print("="*70)
        
        n_observables_range = [2, 3, 4, 5, 6]
        observable_sequence = ['structural', 'electronic', 'thermodynamic', 'magnetic', 'optical', 'structural']
        
        results = {
            'n_observables': [],
            'final_r2_correlated': [],
            'final_r2_independent': []
        }
        
        for n_obs in n_observables_range:
            print(f"\nTesting with {n_obs} observables:")
            print(f"  Types: {observable_sequence[:n_obs]}")
            
            # Correlated
            r2_corr = self.run_with_n_observables(
                n_obs, observable_sequence, correlation_structure='correlated'
            )
            
            # Independent
            r2_indep = self.run_with_n_observables(
                n_obs, observable_sequence, correlation_structure='independent'
            )
            
            results['n_observables'].append(n_obs)
            results['final_r2_correlated'].append(r2_corr[-1])
            results['final_r2_independent'].append(r2_indep[-1])
            
            print(f"  Correlated:   R²={r2_corr[-1]:.4f}")
            print(f"  Independent:  R²={r2_indep[-1]:.4f}")
            print(f"  Benefit of correlation: {(r2_corr[-1] - r2_indep[-1])*100:.2f}%")
        
        return results
    
    def sensitivity_to_observable_choice(self):
        """Test sensitivity to choice of observable types."""
        print("\n" + "="*70)
        print("SENSITIVITY TO OBSERVABLE TYPE CHOICE")
        print("="*70)
        
        # Test different combinations of 3 observables
        combinations = [
            ['structural', 'electronic', 'thermodynamic'],  # Standard physics-based
            ['structural', 'electronic', 'magnetic'],        # Magnetic materials
            ['electronic', 'thermodynamic', 'optical'],      # Semiconductors
            ['structural', 'thermodynamic', 'magnetic'],     # Structural + energetics
            ['structural', 'structural', 'electronic'],      # Repeated (redundant)
        ]
        
        results = {
            'combination': [],
            'final_r2': []
        }
        
        for combo in combinations:
            print(f"\nCombination: {combo}")
            
            r2_scores = self.run_with_n_observables(
                3, combo, correlation_structure='correlated'
            )
            
            results['combination'].append('+'.join(combo))
            results['final_r2'].append(r2_scores[-1])
            
            print(f"  Final R²: {r2_scores[-1]:.4f}")
        
        return results
    
    def run_full_analysis(self):
        """Run complete sensitivity analysis."""
        print("\n" + "="*70)
        print("OBSERVABLE SENSITIVITY ANALYSIS")
        print("="*70)
        
        # Analysis 1: Number of observables
        number_results = self.sensitivity_to_number_of_observables()
        
        # Analysis 2: Choice of observables
        choice_results = self.sensitivity_to_observable_choice()
        
        # Findings
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        optimal_n = number_results['n_observables'][
            np.argmax(number_results['final_r2_correlated'])
        ]
        print(f"\n1. Optimal number of observables: {optimal_n}")
        
        max_r2 = max(number_results['final_r2_correlated'])
        min_r2 = min(number_results['final_r2_correlated'])
        print(f"2. Performance range across n_obs: R²={min_r2:.4f} to {max_r2:.4f}")
        print(f"   Sensitivity: {(max_r2 - min_r2)/min_r2 * 100:.1f}% variation")
        
        corr_benefit = np.mean([
            results['final_r2_correlated'][i] - results['final_r2_independent'][i]
            for i in range(len(number_results['n_observables']))
        ])
        print(f"3. Average benefit of correlation modeling: +{corr_benefit*100:.2f}% R²")
        
        best_combo_idx = np.argmax(choice_results['final_r2'])
        best_combo = choice_results['combination'][best_combo_idx]
        print(f"4. Best observable combination: {best_combo}")
        print(f"   R² = {choice_results['final_r2'][best_combo_idx]:.4f}")
        
        worst_combo_idx = np.argmin(choice_results['final_r2'])
        worst_combo = choice_results['combination'][worst_combo_idx]
        combo_range = choice_results['final_r2'][best_combo_idx] - choice_results['final_r2'][worst_combo_idx]
        print(f"5. Observable choice sensitivity: {combo_range/choice_results['final_r2'][worst_combo_idx] * 100:.1f}% variation")
        
        return {
            'number_sensitivity': number_results,
            'choice_sensitivity': choice_results
        }
    
    def plot_results(self, all_results, save_path='observable_sensitivity_analysis.png'):
        """Visualize sensitivity analysis results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Observable Sensitivity Analysis', fontsize=14, fontweight='bold')
        
        # Plot 1: Number of observables
        ax = axes[0]
        number_results = all_results['number_sensitivity']
        
        x = number_results['n_observables']
        y_corr = number_results['final_r2_correlated']
        y_indep = number_results['final_r2_independent']
        
        ax.plot(x, y_corr, 'o-', label='With Correlation', linewidth=2.5, markersize=9, color='blue')
        ax.plot(x, y_indep, 's--', label='Independent (No Correlation)', linewidth=2, markersize=8, color='orange')
        ax.fill_between(x, y_indep, y_corr, alpha=0.2, color='green', label='Correlation Benefit')
        
        ax.set_xlabel('Number of Observables')
        ax.set_ylabel('Final R² Score')
        ax.set_title('Sensitivity to Number of Observables')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        
        # Plot 2: Observable type choice
        ax = axes[1]
        choice_results = all_results['choice_sensitivity']
        
        combinations = [c.replace('+', '+\n') for c in choice_results['combination']]
        y_vals = choice_results['final_r2']
        
        colors = ['blue' if r == max(y_vals) else 'red' if r == min(y_vals) else 'gray' 
                 for r in y_vals]
        
        bars = ax.bar(range(len(combinations)), y_vals, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(combinations)))
        ax.set_xticklabels(combinations, fontsize=8)
        ax.set_ylabel('Final R² Score')
        ax.set_title('Sensitivity to Observable Type Choice')
        ax.axhline(y=np.mean(y_vals), color='green', linestyle='--', linewidth=2, 
                  label=f'Mean R²={np.mean(y_vals):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    analyzer = ObservableSensitivityAnalysis()
    all_results = analyzer.run_full_analysis()
    
    # Save results
    results_serializable = {
        'number_sensitivity': {
            'n_observables': all_results['number_sensitivity']['n_observables'],
            'final_r2_correlated': [float(x) for x in all_results['number_sensitivity']['final_r2_correlated']],
            'final_r2_independent': [float(x) for x in all_results['number_sensitivity']['final_r2_independent']]
        },
        'choice_sensitivity': {
            'combination': all_results['choice_sensitivity']['combination'],
            'final_r2': [float(x) for x in all_results['choice_sensitivity']['final_r2']]
        }
    }
    
    with open('observable_sensitivity_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("\nResults saved to observable_sensitivity_results.json")
    
    # Plot
    analyzer.plot_results(all_results)
