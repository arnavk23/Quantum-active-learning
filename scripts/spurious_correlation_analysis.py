"""
Spurious Covariance Detection and Failure Analysis

Identifies scenarios where correlated uncertainty in noisy regimes
can mislead the active learning selection strategy.

Shows critical cases: high noise → spurious correlations → wrong sample selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import json


class SpuriousCorrelationDetector:
    """
    Framework to identify and analyze failure modes when correlated
    uncertainty estimates lead to poor sample selection in noisy regimes.
    """
    
    def __init__(self):
        self.noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
        
    def generate_dataset_with_noise(self, n_samples=500, noise_level=0.1, seed=42):
        """Generate synthetic dataset with controlled noise levels."""
        np.random.seed(seed)
        
        # Two correlated properties
        X = np.random.randn(n_samples, 15) * 0.8
        
        # True relationship: simple linear
        y1_true = 2.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1]
        y2_true = 1.0 - 0.4 * X[:, 0] + 0.6 * X[:, 1]
        
        # Add noise - THIS CAN CREATE SPURIOUS CORRELATIONS
        y1 = y1_true + np.random.randn(n_samples) * noise_level
        y2 = y2_true + np.random.randn(n_samples) * noise_level
        
        # Compute empirical correlation from noisy observations
        # In high noise, this may differ significantly from true correlation (-0.4)
        empirical_corr = np.corrcoef(y1, y2)[0, 1]
        
        return X, y1, y2, y1_true, y2_true, empirical_corr
    
    def coupled_uncertainty_with_spurious_correlation(self, model1, model2, X_pool, indices):
        """
        Compute coupled uncertainty using empirical correlation.
        In high noise, empirical correlation can be spurious and misleading.
        """
        uncertainties = []
        
        for idx in indices:
            # Get individual uncertainties
            _, std1 = model1.predict(X_pool[idx:idx+1], return_std=True)
            _, std2 = model2.predict(X_pool[idx:idx+1], return_std=True)
            
            # Empirical correlation (can be spurious in high noise)
            # This is what the algorithm "sees"
            corr_empirical = np.corrcoef(model1.y_train_, model2.y_train_)[0, 1]
            
            # Coupled uncertainty (assuming correlation helps reduce uncertainty)
            # BUG: In high noise, spurious correlation can mislead this calculation
            coupled_unc = np.sqrt(
                std1[0]**2 + std2[0]**2 + 2 * corr_empirical * std1[0] * std2[0]
            )
            uncertainties.append(coupled_unc)
        
        return np.array(uncertainties), corr_empirical
    
    def true_covariance_analysis(self, y1_true, y2_true):
        """Analyze true (noise-free) covariance structure."""
        true_corr = np.corrcoef(y1_true, y2_true)[0, 1]
        return true_corr
    
    def run_noise_robustness_experiment(self, n_iterations=6, batch_size=10, initial_pool_size=40):
        """
        Run experiment across noise levels to show when spurious correlations
        cause selection failures.
        """
        print("=" * 80)
        print("Spurious Covariance Detection Experiment")
        print("=" * 80)
        
        results_by_noise = {}
        
        for noise_level in self.noise_levels:
            print(f"\nNoise Level: {noise_level:.3f}")
            print("-" * 80)
            
            X_train, y1_train, y2_train, y1_true, y2_true, corr_empirical = \
                self.generate_dataset_with_noise(n_samples=400, noise_level=noise_level)
            
            X_test, y1_test, y2_test, _, _, _ = \
                self.generate_dataset_with_noise(n_samples=100, noise_level=noise_level)
            
            # Get true correlation
            true_corr = self.true_covariance_analysis(y1_true, y2_true)
            
            print(f"  True correlation: {true_corr:.4f}")
            print(f"  Empirical correlation (from noisy data): {corr_empirical:.4f}")
            print(f"  Spurious correlation error: {abs(corr_empirical - true_corr):.4f}")
            
            # Initialize labeled set
            labeled_idx = np.random.choice(len(X_train), size=initial_pool_size, replace=False)
            pool_idx = np.array([i for i in range(len(X_train)) if i not in labeled_idx])
            
            iteration_results = {
                'iteration': [],
                'r2_y1': [],
                'r2_y2': [],
                'correlation_agreement': [],  # How well empirical matches true
                'mse_y1': [],
                'mse_y2': []
            }
            
            for iteration in range(n_iterations):
                # Train GP models for both properties
                model1 = GaussianProcessRegressor(
                    kernel=Matern(nu=2.5),
                    n_restarts_optimizer=5,
                    alpha=1e-6,
                    normalize_y=True
                )
                model2 = GaussianProcessRegressor(
                    kernel=Matern(nu=2.5),
                    n_restarts_optimizer=5,
                    alpha=1e-6,
                    normalize_y=True
                )
                
                model1.fit(X_train[labeled_idx], y1_train[labeled_idx])
                model2.fit(X_train[labeled_idx], y2_train[labeled_idx])
                
                # Evaluate
                y1_pred = model1.predict(X_test)
                y2_pred = model2.predict(X_test)
                
                r2_y1 = r2_score(y1_test, y1_pred)
                r2_y2 = r2_score(y2_test, y2_pred)
                mse_y1 = mean_absolute_error(y1_test, y1_pred)
                mse_y2 = mean_absolute_error(y2_test, y2_pred)
                
                # Compute agreement between empirical and true correlation
                empirical_corr = np.corrcoef(model1.y_train_, model2.y_train_)[0, 1]
                corr_agreement = 1.0 - abs(empirical_corr - true_corr)
                
                iteration_results['iteration'].append(iteration + 1)
                iteration_results['r2_y1'].append(r2_y1)
                iteration_results['r2_y2'].append(r2_y2)
                iteration_results['correlation_agreement'].append(corr_agreement)
                iteration_results['mse_y1'].append(mse_y1)
                iteration_results['mse_y2'].append(mse_y2)
                
                print(f"  Iter {iteration+1}: R²_y1={r2_y1:.4f}, R²_y2={r2_y2:.4f}, " \
                      f"CorrAgreement={corr_agreement:.4f}")
                
                # Query next batch
                if len(pool_idx) > 0:
                    uncertainties, _ = self.coupled_uncertainty_with_spurious_correlation(
                        model1, model2, X_train, pool_idx
                    )
                    
                    top_indices = np.argsort(uncertainties)[-batch_size:]
                    selected_samples = pool_idx[top_indices]
                    
                    labeled_idx = np.concatenate([labeled_idx, selected_samples])
                    pool_idx = np.array([i for i in pool_idx if i not in selected_samples])
            
            results_by_noise[f'{noise_level:.2f}'] = {
                'true_corr': float(true_corr),
                'empirical_corr': float(corr_empirical),
                'results': iteration_results
            }
        
        return results_by_noise
    
    def identify_failure_scenarios(self, results_by_noise):
        """
        Analyze results to identify scenarios where spurious correlations
        cause active learning failures.
        """
        print("\n" + "=" * 80)
        print("FAILURE SCENARIO ANALYSIS")
        print("=" * 80)
        
        failure_scenarios = []
        
        for noise_str, data in results_by_noise.items():
            noise_level = float(noise_str)
            true_corr = data['true_corr']
            empirical_corr = data['empirical_corr']
            results = data['results']
            
            # Metrics for failure detection
            final_r2_y1 = results['r2_y1'][-1]
            final_r2_y2 = results['r2_y2'][-1]
            avg_corr_agreement = np.mean(results['correlation_agreement'])
            
            corr_error = abs(empirical_corr - true_corr)
            
            # Failure criterion: high noise + high correlation error + low final R²
            is_failure = (noise_level > 0.1 and corr_error > 0.15 and 
                         (final_r2_y1 < 0.6 or final_r2_y2 < 0.6))
            
            if is_failure or noise_level >= 0.1:  # Report high-noise cases
                scenario = {
                    'noise_level': noise_level,
                    'true_correlation': true_corr,
                    'empirical_correlation': empirical_corr,
                    'correlation_error': corr_error,
                    'final_r2_y1': final_r2_y1,
                    'final_r2_y2': final_r2_y2,
                    'avg_corr_agreement': avg_corr_agreement,
                    'is_failure': is_failure,
                    'recommendation': (
                        'Use robust correlation estimation (e.g., Spearman) or '
                        'reduce coupling weight in high-noise regime' if is_failure 
                        else 'Monitor correlation estimation quality'
                    )
                }
                failure_scenarios.append(scenario)
                
                if is_failure:
                    print(f"\n⚠️  FAILURE DETECTED at Noise Level {noise_level:.3f}:")
                else:
                    print(f"\n⚠️  HIGH-RISK scenario at Noise Level {noise_level:.3f}:")
                
                print(f"   - Correlation Error: {corr_error:.4f} (TRUE={true_corr:.4f}, EMPIRICAL={empirical_corr:.4f})")
                print(f"   - Final R²: y1={final_r2_y1:.4f}, y2={final_r2_y2:.4f}")
                print(f"   - Avg Correlation Agreement: {avg_corr_agreement:.4f}")
                print(f"   - Recommendation: {scenario['recommendation']}")
        
        return failure_scenarios
    
    def plot_spurious_correlation_analysis(self, results_by_noise, save_path='spurious_correlation_analysis.png'):
        """Visualize spurious correlation effects."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Spurious Correlation Analysis: Failure Modes in Noisy Regimes', 
                     fontsize=14, fontweight='bold')
        
        noise_levels = [float(k) for k in sorted(results_by_noise.keys())]
        
        # Track metrics across noise levels
        true_corrs = []
        empirical_corrs = []
        final_r2_y1_list = []
        final_r2_y2_list = []
        
        for noise_str in sorted(results_by_noise.keys()):
            data = results_by_noise[noise_str]
            true_corrs.append(data['true_corr'])
            empirical_corrs.append(data['empirical_corr'])
            final_r2_y1_list.append(data['results']['r2_y1'][-1])
            final_r2_y2_list.append(data['results']['r2_y2'][-1])
        
        # Plot 1: Correlation Error across Noise Levels
        ax = axes[0, 0]
        corr_errors = [abs(e - t) for e, t in zip(empirical_corrs, true_corrs)]
        ax.plot(noise_levels, corr_errors, 'o-', linewidth=2.5, markersize=8, color='red')
        ax.axhline(y=0.15, color='orange', linestyle='--', linewidth=2, label='Failure Threshold')
        ax.fill_between(noise_levels, 0, corr_errors, alpha=0.3, color='red')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Correlation Estimation Error')
        ax.set_title('Spurious Correlation Growth with Noise')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Empirical vs True Correlation
        ax = axes[0, 1]
        ax.plot(noise_levels, true_corrs, 'o-', linewidth=2.5, markersize=8, 
               label='True Correlation', color='blue')
        ax.plot(noise_levels, empirical_corrs, 's--', linewidth=2.5, markersize=8, 
               label='Empirical Correlation (Noisy)', color='red')
        ax.fill_between(noise_levels, true_corrs, empirical_corrs, 
                       alpha=0.2, color='purple')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Correlation Value')
        ax.set_title('True vs Empirical Correlations')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Final R² Scores
        ax = axes[1, 0]
        ax.plot(noise_levels, final_r2_y1_list, 'o-', linewidth=2.5, markersize=8, 
               label='Property Y₁', color='green')
        ax.plot(noise_levels, final_r2_y2_list, 's--', linewidth=2.5, markersize=8, 
               label='Property Y₂', color='purple')
        ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Failure Threshold (R²=0.6)')
        ax.fill_between(noise_levels, 0, final_r2_y1_list, alpha=0.1, color='green')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Final R² Score')
        ax.set_title('Model Performance Degradation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Failure Region Heatmap
        ax = axes[1, 1]
        failure_scores = np.array([
            (1 - (r2_y1 + r2_y2) / 2) * (1 + corr_err)
            for r2_y1, r2_y2, corr_err in zip(final_r2_y1_list, final_r2_y2_list, 
                                             [abs(e - t) for e, t in zip(empirical_corrs, true_corrs)])
        ])
        
        colors = ['red' if score > 0.5 else 'orange' if score > 0.3 else 'green' 
                 for score in failure_scores]
        ax.bar(range(len(noise_levels)), failure_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f'{nl:.2f}' for nl in noise_levels])
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Failure Score')
        ax.set_title('Failure Risk Index\n(Red=High Risk, Orange=Moderate, Green=Safe)')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    detector = SpuriousCorrelationDetector()
    results_by_noise = detector.run_noise_robustness_experiment(n_iterations=6, batch_size=10)
    
    # Analyze failures
    failure_scenarios = detector.identify_failure_scenarios(results_by_noise)
    
    # Save results
    with open('spurious_correlation_results.json', 'w') as f:
        results_serializable = {}
        for noise_str, data in results_by_noise.items():
            results_serializable[noise_str] = {
                'true_corr': float(data['true_corr']),
                'empirical_corr': float(data['empirical_corr']),
                'iteration': [int(x) for x in data['results']['iteration']],
                'r2_y1': [float(x) for x in data['results']['r2_y1']],
                'r2_y2': [float(x) for x in data['results']['r2_y2']],
                'correlation_agreement': [float(x) for x in data['results']['correlation_agreement']],
                'mse_y1': [float(x) for x in data['results']['mse_y1']],
                'mse_y2': [float(x) for x in data['results']['mse_y2']]
            }
        json.dump(results_serializable, f, indent=2)
    
    print("\nDetailed results saved to spurious_correlation_results.json")
    
    # Plot
    detector.plot_spurious_correlation_analysis(results_by_noise)
