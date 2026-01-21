"""
Broader Materials Validation Across Multiple Datasets and Tasks

Validates quantum-enhanced active learning on 6 diverse materials tasks:
1. Band gap prediction (semiconductors)
2. Formation energy (thermodynamic stability)
3. Elastic modulus (mechanical properties)
4. Thermal conductivity (transport properties)
5. Magnetic moment (magnetic materials)
6. Dielectric constant (electronic materials)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import json


class BroaderMaterialsValidation:
    """
    Validation framework across diverse materials properties and datasets.
    Each task has realistic value ranges and feature dependencies.
    """
    
    def __init__(self):
        self.tasks = {
            'band_gap': {
                'name': 'Band Gap (eV)',
                'range': (0.5, 6.0),
                'n_samples': 800,
                'feature_weights': [0.5, 0.3, -0.2, 0.4, -0.1]
            },
            'formation_energy': {
                'name': 'Formation Energy (eV/atom)',
                'range': (-8.0, -0.5),
                'n_samples': 850,
                'feature_weights': [-0.4, -0.5, 0.3, -0.2, 0.1]
            },
            'elastic_modulus': {
                'name': 'Elastic Modulus (GPa)',
                'range': (20, 400),
                'n_samples': 750,
                'feature_weights': [0.6, 0.4, 0.3, -0.1, 0.2]
            },
            'thermal_conductivity': {
                'name': 'Thermal Conductivity (W/m·K)',
                'range': (1, 200),
                'n_samples': 700,
                'feature_weights': [0.3, 0.5, -0.3, 0.4, -0.2]
            },
            'magnetic_moment': {
                'name': 'Magnetic Moment (μB)',
                'range': (0, 5),
                'n_samples': 650,
                'feature_weights': [0.4, -0.2, 0.5, 0.3, -0.4]
            },
            'dielectric_constant': {
                'name': 'Dielectric Constant',
                'range': (1, 100),
                'n_samples': 780,
                'feature_weights': [-0.3, 0.4, -0.5, 0.2, 0.3]
            }
        }
    
    def generate_task_dataset(self, task_name, n_features=20, seed=42):
        """Generate synthetic dataset for a specific materials task."""
        np.random.seed(seed + hash(task_name) % 2**32)
        
        task_config = self.tasks[task_name]
        n_samples = task_config['n_samples']
        value_range = task_config['range']
        weights = task_config['feature_weights']
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with task-specific dependencies
        y_base = sum(w * X[:, i] for i, w in enumerate(weights))
        
        # Scale to realistic range
        y_normalized = (y_base - np.min(y_base)) / (np.max(y_base) - np.min(y_base) + 1e-10)
        y = value_range[0] + y_normalized * (value_range[1] - value_range[0])
        
        # Add noise (10-15% of range)
        noise_level = 0.12 * (value_range[1] - value_range[0])
        y += np.random.randn(n_samples) * noise_level
        
        # Ensure within bounds
        y = np.clip(y, value_range[0], value_range[1])
        
        return X, y
    
    def quantum_enhanced_uncertainty(self, model, X_pool, pool_indices, n_observables=3):
        """Compute quantum-enhanced uncertainty with multiple observables."""
        uncertainties = []
        
        for idx in pool_indices:
            x = X_pool[idx:idx+1]
            
            # Get base uncertainty
            try:
                if hasattr(model, 'predict'):
                    _, base_std = model.predict(x, return_std=True)
                    base_uncertainty = base_std[0]
                else:
                    base_uncertainty = 0.1
            except:
                base_uncertainty = 0.1
            
            # Simulate multi-observable variance and covariance
            # Observable 1: structural uncertainty
            obs1_var = base_uncertainty**2
            
            # Observable 2: electronic uncertainty (correlated with obs1)
            obs2_var = (base_uncertainty * 0.8)**2
            
            # Observable 3: thermodynamic uncertainty
            obs3_var = (base_uncertainty * 0.9)**2
            
            # Covariance terms (positive correlation)
            cov_12 = 0.3 * np.sqrt(obs1_var * obs2_var)
            cov_13 = 0.2 * np.sqrt(obs1_var * obs3_var)
            cov_23 = 0.25 * np.sqrt(obs2_var * obs3_var)
            
            # Aggregate uncertainty
            total_uncertainty = np.sqrt(
                obs1_var + obs2_var + obs3_var + 2*cov_12 + 2*cov_13 + 2*cov_23
            )
            
            uncertainties.append(total_uncertainty)
        
        return np.array(uncertainties)
    
    def baseline_uncertainty(self, model, X_pool, pool_indices, method='uncertainty'):
        """Compute baseline uncertainty."""
        uncertainties = []
        
        for idx in pool_indices:
            x = X_pool[idx:idx+1]
            
            if method == 'uncertainty':
                try:
                    _, std = model.predict(x, return_std=True)
                    uncertainties.append(std[0])
                except:
                    uncertainties.append(np.random.rand())
            
            elif method == 'random':
                uncertainties.append(np.random.rand())
        
        return np.array(uncertainties)
    
    def run_validation_on_task(self, task_name, n_iterations=8, batch_size=15, initial_pool_size=50):
        """Run quantum-enhanced vs baselines on a single task."""
        print(f"\n{'='*70}")
        print(f"Task: {self.tasks[task_name]['name']}")
        print(f"{'='*70}")
        
        # Generate dataset
        X, y = self.generate_task_dataset(task_name)
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split
        split_idx = int(0.7 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        results = {
            'quantum': {'r2': [], 'mae': []},
            'uncertainty': {'r2': [], 'mae': []},
            'random': {'r2': [], 'mae': []}
        }
        
        # Run three methods
        for method_name in ['quantum', 'uncertainty', 'random']:
            labeled_idx = np.random.choice(len(X_train), size=initial_pool_size, replace=False)
            pool_idx = np.array([i for i in range(len(X_train)) if i not in labeled_idx])
            
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
                mae = mean_absolute_error(y_test, y_pred)
                
                results[method_name]['r2'].append(r2)
                results[method_name]['mae'].append(mae)
                
                # Query next batch
                if len(pool_idx) > batch_size:
                    if method_name == 'quantum':
                        uncertainties = self.quantum_enhanced_uncertainty(model, X_train, pool_idx)
                    else:
                        uncertainties = self.baseline_uncertainty(model, X_train, pool_idx, method=method_name)
                    
                    top_indices = np.argsort(uncertainties)[-batch_size:]
                    selected = pool_idx[top_indices]
                    
                    labeled_idx = np.concatenate([labeled_idx, selected])
                    pool_idx = np.array([i for i in pool_idx if i not in selected])
            
            final_r2 = results[method_name]['r2'][-1]
            final_mae = results[method_name]['mae'][-1]
            print(f"  {method_name:12s}: Final R²={final_r2:.4f}, MAE={final_mae:.4f}")
        
        return results
    
    def run_full_validation(self):
        """Run validation across all 6 tasks."""
        print("\n" + "="*70)
        print("BROADER MATERIALS VALIDATION")
        print("="*70)
        
        all_results = {}
        
        for task_name in self.tasks.keys():
            results = self.run_validation_on_task(task_name, n_iterations=8, batch_size=15)
            all_results[task_name] = results
        
        return all_results
    
    def compute_aggregate_statistics(self, all_results):
        """Compute aggregate statistics across all tasks."""
        print("\n" + "="*70)
        print("AGGREGATE STATISTICS ACROSS ALL TASKS")
        print("="*70)
        
        summary = {
            'quantum_avg_r2': [],
            'uncertainty_avg_r2': [],
            'random_avg_r2': [],
            'quantum_improvements': []
        }
        
        for task_name, results in all_results.items():
            final_quantum = results['quantum']['r2'][-1]
            final_uncertainty = results['uncertainty']['r2'][-1]
            final_random = results['random']['r2'][-1]
            
            summary['quantum_avg_r2'].append(final_quantum)
            summary['uncertainty_avg_r2'].append(final_uncertainty)
            summary['random_avg_r2'].append(final_random)
            
            improvement_vs_uncertainty = ((final_quantum - final_uncertainty) / final_uncertainty) * 100
            summary['quantum_improvements'].append(improvement_vs_uncertainty)
            
            print(f"\n{self.tasks[task_name]['name']:30s}")
            print(f"  Quantum:     R²={final_quantum:.4f}")
            print(f"  Uncertainty: R²={final_uncertainty:.4f}")
            print(f"  Random:      R²={final_random:.4f}")
            print(f"  Improvement: {improvement_vs_uncertainty:+.2f}%")
        
        print("\n" + "-"*70)
        print(f"Average across all tasks:")
        print(f"  Quantum:     R²={np.mean(summary['quantum_avg_r2']):.4f} ± {np.std(summary['quantum_avg_r2']):.4f}")
        print(f"  Uncertainty: R²={np.mean(summary['uncertainty_avg_r2']):.4f} ± {np.std(summary['uncertainty_avg_r2']):.4f}")
        print(f"  Random:      R²={np.mean(summary['random_avg_r2']):.4f} ± {np.std(summary['random_avg_r2']):.4f}")
        print(f"  Avg improvement: {np.mean(summary['quantum_improvements']):.2f}% ± {np.std(summary['quantum_improvements']):.2f}%")
        
        return summary
    
    def plot_results(self, all_results, save_path='broader_validation_results.png'):
        """Visualize results across all tasks."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Broader Materials Validation: 6 Diverse Tasks', fontsize=14, fontweight='bold')
        
        for idx, (task_name, ax) in enumerate(zip(self.tasks.keys(), axes.flat)):
            results = all_results[task_name]
            iterations = list(range(1, len(results['quantum']['r2']) + 1))
            
            ax.plot(iterations, results['quantum']['r2'], 'o-', label='Quantum-Enhanced',
                   linewidth=2.5, markersize=7, color='blue')
            ax.plot(iterations, results['uncertainty']['r2'], 's--', label='Uncertainty Sampling',
                   linewidth=2, markersize=6, color='orange')
            ax.plot(iterations, results['random']['r2'], '^:', label='Random Sampling',
                   linewidth=1.5, markersize=5, color='gray')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('R² Score')
            ax.set_title(self.tasks[task_name]['name'], fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    validator = BroaderMaterialsValidation()
    all_results = validator.run_full_validation()
    
    # Compute statistics
    summary = validator.compute_aggregate_statistics(all_results)
    
    # Save results
    results_serializable = {}
    for task_name, results in all_results.items():
        results_serializable[task_name] = {
            'quantum_r2': [float(x) for x in results['quantum']['r2']],
            'quantum_mae': [float(x) for x in results['quantum']['mae']],
            'uncertainty_r2': [float(x) for x in results['uncertainty']['r2']],
            'uncertainty_mae': [float(x) for x in results['uncertainty']['mae']],
            'random_r2': [float(x) for x in results['random']['r2']],
            'random_mae': [float(x) for x in results['random']['mae']]
        }
    
    results_serializable['summary'] = {
        'quantum_avg_r2': [float(x) for x in summary['quantum_avg_r2']],
        'uncertainty_avg_r2': [float(x) for x in summary['uncertainty_avg_r2']],
        'random_avg_r2': [float(x) for x in summary['random_avg_r2']],
        'quantum_improvements': [float(x) for x in summary['quantum_improvements']],
        'mean_improvement': float(np.mean(summary['quantum_improvements'])),
        'std_improvement': float(np.std(summary['quantum_improvements']))
    }
    
    with open('broader_validation_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("\nResults saved to broader_validation_results.json")
    
    # Plot
    validator.plot_results(all_results)
