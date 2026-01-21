"""
Pareto-Based vs Quantum-Enhanced Multi-Objective Active Learning

Compares the quantum-enhanced scalar aggregation approach with 
Pareto frontier-based multi-objective active learning methods.

Metrics: Pareto hypervolume, runtime, sample efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import time
import json


class ParetoVsQuantumComparison:
    """
    Compare Pareto-based selection with quantum-enhanced scalar aggregation
    for multi-objective materials optimization.
    """
    
    def __init__(self):
        self.n_objectives = 3
        
    def generate_multi_objective_dataset(self, n_samples=600, seed=42):
        """Generate dataset with 3 correlated objectives."""
        np.random.seed(seed)
        
        # Features
        X = np.random.randn(n_samples, 15)
        
        # Three objectives with realistic correlations
        # Objective 1: Band gap (maximize)
        y1 = 2.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n_samples) * 0.2
        
        # Objective 2: Formation energy (minimize, correlated with band gap)
        y2 = -3.0 - 0.4 * X[:, 0] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.3
        
        # Objective 3: Thermal conductivity (maximize, correlated with both)
        y3 = 50 + 20 * X[:, 1] - 10 * X[:, 2] + np.random.randn(n_samples) * 5
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, np.column_stack([y1, y2, y3])
    
    def is_pareto_efficient(self, costs):
        """
        Find Pareto-efficient points (minimization).
        For maximization objectives, negate before passing.
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated points
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient
    
    def compute_hypervolume(self, pareto_front, reference_point):
        """
        Compute hypervolume indicator (approximation for 3D).
        Higher hypervolume = better coverage of objective space.
        """
        if len(pareto_front) == 0:
            return 0.0
        
        # Simple approximation: sum of dominated volumes
        hypervolume = 0.0
        for point in pareto_front:
            volume = np.prod(np.maximum(reference_point - point, 0))
            hypervolume += volume
        
        return hypervolume
    
    def pareto_based_selection(self, X_pool, models, pool_indices, batch_size=10):
        """
        Pareto-based active learning: select samples on predicted Pareto frontier.
        Runtime: O(n^2 log n) for n pool candidates.
        """
        start_time = time.time()
        
        # Predict all objectives for pool
        predictions = []
        uncertainties = []
        
        for model in models:
            pred, std = model.predict(X_pool[pool_indices], return_std=True)
            predictions.append(pred)
            uncertainties.append(std)
        
        predictions = np.array(predictions).T  # Shape: (n_pool, n_objectives)
        uncertainties = np.array(uncertainties).T
        
        # Convert to minimization (negate maximization objectives)
        # Assume objectives: [maximize y1, minimize y2, maximize y3]
        costs = predictions.copy()
        costs[:, 0] = -costs[:, 0]  # y1: maximize -> minimize
        costs[:, 2] = -costs[:, 2]  # y3: maximize -> minimize
        
        # Find Pareto frontier
        is_pareto = self.is_pareto_efficient(costs)
        pareto_indices = np.where(is_pareto)[0]
        
        # Among Pareto points, select those with highest uncertainty
        if len(pareto_indices) >= batch_size:
            # Sum uncertainties across objectives
            total_uncertainties = np.sum(uncertainties[pareto_indices], axis=1)
            top_in_pareto = np.argsort(total_uncertainties)[-batch_size:]
            selected_local = pareto_indices[top_in_pareto]
        else:
            # Pareto frontier too small, add high-uncertainty points
            selected_local = list(pareto_indices)
            remaining = batch_size - len(selected_local)
            total_uncertainties = np.sum(uncertainties, axis=1)
            non_pareto_indices = np.where(~is_pareto)[0]
            if len(non_pareto_indices) > 0:
                top_uncertain = np.argsort(total_uncertainties[non_pareto_indices])[-remaining:]
                selected_local.extend(non_pareto_indices[top_uncertain])
            selected_local = np.array(selected_local)
        
        runtime = time.time() - start_time
        selected_global = pool_indices[selected_local]
        
        return selected_global, runtime
    
    def quantum_scalar_selection(self, X_pool, models, pool_indices, batch_size=10):
        """
        Quantum-enhanced scalar aggregation: couple uncertainties via covariance.
        Runtime: O(n) for n pool candidates.
        """
        start_time = time.time()
        
        # Predict with uncertainties
        predictions = []
        uncertainties = []
        
        for model in models:
            pred, std = model.predict(X_pool[pool_indices], return_std=True)
            predictions.append(pred)
            uncertainties.append(std)
        
        uncertainties = np.array(uncertainties).T  # Shape: (n_pool, n_objectives)
        
        # Aggregate with covariance coupling
        # Assume positive correlations between objectives
        correlation_matrix = np.array([
            [1.0,  0.3,  0.2],
            [0.3,  1.0,  0.25],
            [0.2,  0.25, 1.0]
        ])
        
        # Compute coupled uncertainty for each candidate
        coupled_uncertainties = []
        for unc_vec in uncertainties:
            total_unc = np.sqrt(unc_vec @ correlation_matrix @ unc_vec)
            coupled_uncertainties.append(total_unc)
        
        coupled_uncertainties = np.array(coupled_uncertainties)
        
        # Select top batch
        top_indices = np.argsort(coupled_uncertainties)[-batch_size:]
        selected_global = pool_indices[top_indices]
        
        runtime = time.time() - start_time
        
        return selected_global, runtime
    
    def run_comparison(self, n_iterations=6, batch_size=10, initial_pool_size=30):
        """Run comparative experiment: Pareto vs Quantum."""
        print("\n" + "="*70)
        print("Pareto-Based vs Quantum-Enhanced Multi-Objective Active Learning")
        print("="*70)
        
        # Generate dataset
        X, Y = self.generate_multi_objective_dataset(n_samples=600)
        
        # Split
        split_idx = int(0.7 * len(X))
        X_train, Y_train = X[:split_idx], Y[:split_idx]
        X_test, Y_test = X[split_idx:], Y[split_idx:]
        
        results = {
            'pareto': {
                'r2_per_objective': [[] for _ in range(self.n_objectives)],
                'hypervolume': [],
                'runtime_per_iter': []
            },
            'quantum': {
                'r2_per_objective': [[] for _ in range(self.n_objectives)],
                'hypervolume': [],
                'runtime_per_iter': []
            }
        }
        
        # Reference point for hypervolume (worst case)
        reference_point = np.array([-10, 5, 0])  # Adjusted for cost space
        
        # Run both methods
        for method_name, selection_func in [('pareto', self.pareto_based_selection),
                                            ('quantum', self.quantum_scalar_selection)]:
            print(f"\n{method_name.upper()} Method:")
            print("-" * 70)
            
            labeled_idx = np.random.choice(len(X_train), size=initial_pool_size, replace=False)
            pool_idx = np.array([i for i in range(len(X_train)) if i not in labeled_idx])
            
            for iteration in range(n_iterations):
                # Train models for each objective
                models = []
                r2_scores = []
                
                for obj_idx in range(self.n_objectives):
                    model = GaussianProcessRegressor(
                        kernel=Matern(nu=2.5),
                        n_restarts_optimizer=3,
                        alpha=1e-6,
                        normalize_y=True,
                        random_state=42
                    )
                    model.fit(X_train[labeled_idx], Y_train[labeled_idx, obj_idx])
                    models.append(model)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    r2 = r2_score(Y_test[:, obj_idx], y_pred)
                    r2_scores.append(r2)
                    results[method_name]['r2_per_objective'][obj_idx].append(r2)
                
                # Compute hypervolume on test set
                test_predictions = np.array([
                    model.predict(X_test) for model in models
                ]).T
                
                # Convert to costs
                test_costs = test_predictions.copy()
                test_costs[:, 0] = -test_costs[:, 0]
                test_costs[:, 2] = -test_costs[:, 2]
                
                # Find Pareto front on test predictions
                is_pareto_test = self.is_pareto_efficient(test_costs)
                pareto_front_test = test_costs[is_pareto_test]
                hypervolume = self.compute_hypervolume(pareto_front_test, reference_point)
                results[method_name]['hypervolume'].append(hypervolume)
                
                print(f"  Iter {iteration+1}: R²=[{r2_scores[0]:.3f}, {r2_scores[1]:.3f}, {r2_scores[2]:.3f}], HV={hypervolume:.2f}", end='')
                
                # Query next batch
                if len(pool_idx) > batch_size:
                    selected, runtime = selection_func(X_train, models, pool_idx, batch_size)
                    results[method_name]['runtime_per_iter'].append(runtime)
                    print(f", Time={runtime:.4f}s")
                    
                    labeled_idx = np.concatenate([labeled_idx, selected])
                    pool_idx = np.array([i for i in pool_idx if i not in selected])
                else:
                    print()
        
        return results
    
    def plot_comparison(self, results, save_path='pareto_vs_quantum_comparison.png'):
        """Visualize comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pareto vs Quantum: Multi-Objective Active Learning', 
                     fontsize=14, fontweight='bold')
        
        iterations = list(range(1, len(results['pareto']['hypervolume']) + 1))
        
        # Plot 1: Hypervolume comparison
        ax = axes[0, 0]
        ax.plot(iterations, results['pareto']['hypervolume'], 'o-', 
               label='Pareto-Based', linewidth=2.5, markersize=8, color='blue')
        ax.plot(iterations, results['quantum']['hypervolume'], 's--', 
               label='Quantum Scalar', linewidth=2.5, markersize=8, color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypervolume Indicator')
        ax.set_title('Coverage of Objective Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Runtime comparison
        ax = axes[0, 1]
        pareto_runtime = results['pareto']['runtime_per_iter']
        quantum_runtime = results['quantum']['runtime_per_iter']
        
        ax.plot(iterations[:len(pareto_runtime)], pareto_runtime, 'o-', 
               label='Pareto-Based', linewidth=2.5, markersize=8, color='blue')
        ax.plot(iterations[:len(quantum_runtime)], quantum_runtime, 's--', 
               label='Quantum Scalar', linewidth=2.5, markersize=8, color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Selection Time (seconds)')
        ax.set_title('Computational Cost per Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 3: Average R² across objectives
        ax = axes[1, 0]
        pareto_avg_r2 = [np.mean([results['pareto']['r2_per_objective'][j][i] 
                                  for j in range(3)]) for i in range(len(iterations))]
        quantum_avg_r2 = [np.mean([results['quantum']['r2_per_objective'][j][i] 
                                   for j in range(3)]) for i in range(len(iterations))]
        
        ax.plot(iterations, pareto_avg_r2, 'o-', 
               label='Pareto-Based', linewidth=2.5, markersize=8, color='blue')
        ax.plot(iterations, quantum_avg_r2, 's--', 
               label='Quantum Scalar', linewidth=2.5, markersize=8, color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average R² Across Objectives')
        ax.set_title('Predictive Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Hypervolume per unit time (efficiency)
        ax = axes[1, 1]
        if len(pareto_runtime) > 0 and len(quantum_runtime) > 0:
            pareto_efficiency = [hv / max(rt, 0.001) for hv, rt in 
                                zip(results['pareto']['hypervolume'][:len(pareto_runtime)], 
                                    pareto_runtime)]
            quantum_efficiency = [hv / max(rt, 0.001) for hv, rt in 
                                 zip(results['quantum']['hypervolume'][:len(quantum_runtime)], 
                                     quantum_runtime)]
            
            ax.plot(iterations[:len(pareto_efficiency)], pareto_efficiency, 'o-', 
                   label='Pareto-Based', linewidth=2.5, markersize=8, color='blue')
            ax.plot(iterations[:len(quantum_efficiency)], quantum_efficiency, 's--', 
                   label='Quantum Scalar', linewidth=2.5, markersize=8, color='orange')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Hypervolume / Time')
            ax.set_title('Computational Efficiency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    comparison = ParetoVsQuantumComparison()
    results = comparison.run_comparison(n_iterations=6, batch_size=10)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    pareto_final_hv = results['pareto']['hypervolume'][-1]
    quantum_final_hv = results['quantum']['hypervolume'][-1]
    hv_diff_pct = ((pareto_final_hv - quantum_final_hv) / quantum_final_hv) * 100
    
    pareto_avg_time = np.mean(results['pareto']['runtime_per_iter'])
    quantum_avg_time = np.mean(results['quantum']['runtime_per_iter'])
    time_ratio = pareto_avg_time / quantum_avg_time
    
    print(f"\nFinal Hypervolume:")
    print(f"  Pareto:  {pareto_final_hv:.2f}")
    print(f"  Quantum: {quantum_final_hv:.2f}")
    print(f"  Difference: {hv_diff_pct:+.1f}%")
    
    print(f"\nAverage Runtime per Iteration:")
    print(f"  Pareto:  {pareto_avg_time:.4f}s")
    print(f"  Quantum: {quantum_avg_time:.4f}s")
    print(f"  Speedup: {time_ratio:.1f}x slower for Pareto")
    
    # Save results
    results_serializable = {
        'pareto': {
            'r2_per_objective': [[float(x) for x in obj_scores] 
                                for obj_scores in results['pareto']['r2_per_objective']],
            'hypervolume': [float(x) for x in results['pareto']['hypervolume']],
            'runtime_per_iter': [float(x) for x in results['pareto']['runtime_per_iter']]
        },
        'quantum': {
            'r2_per_objective': [[float(x) for x in obj_scores] 
                                for obj_scores in results['quantum']['r2_per_objective']],
            'hypervolume': [float(x) for x in results['quantum']['hypervolume']],
            'runtime_per_iter': [float(x) for x in results['quantum']['runtime_per_iter']]
        },
        'summary': {
            'pareto_final_hv': float(pareto_final_hv),
            'quantum_final_hv': float(quantum_final_hv),
            'hv_difference_pct': float(hv_diff_pct),
            'pareto_avg_time': float(pareto_avg_time),
            'quantum_avg_time': float(quantum_avg_time),
            'time_ratio': float(time_ratio)
        }
    }
    
    with open('pareto_quantum_comparison.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("\nResults saved to pareto_quantum_comparison.json")
    
    # Plot
    comparison.plot_comparison(results)
