"""
Discrete Materials Classification using Quantum-Enhanced Active Learning

Extends framework to discrete classification tasks (crystal systems, stability classes)
beyond continuous property regression, demonstrating broader applicability.

Classification targets: Crystal system (cubic, hexagonal, tetragonal, etc.),
Stability class (stable, metastable, unstable)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import json


class DiscreteClassificationFramework:
    """
    Quantum-enhanced active learning for discrete materials classification.
    Extends continuous property prediction to discrete categorical tasks.
    """
    
    def __init__(self, n_classes=6):
        self.n_classes = n_classes
        self.class_names = ['Cubic', 'Hexagonal', 'Tetragonal', 'Orthorhombic', 'Monoclinic', 'Triclinic']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        
    def generate_synthetic_classification_dataset(self, n_samples=1200, seed=42):
        """
        Generate synthetic crystallographic classification dataset
        with class-dependent feature distributions.
        """
        np.random.seed(seed)
        n_features = 20
        
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)
        
        samples_per_class = n_samples // self.n_classes
        
        for class_idx in range(self.n_classes):
            start_idx = class_idx * samples_per_class
            end_idx = start_idx + samples_per_class
            
            # Generate class-specific feature distributions
            # Cubic-like features: symmetric
            if class_idx == 0:
                X[start_idx:end_idx] = np.random.randn(samples_per_class, n_features) * 0.8 + 2.0
            # Hexagonal-like features: anisotropic
            elif class_idx == 1:
                X[start_idx:end_idx, :10] = np.random.randn(samples_per_class, 10) * 1.2 + 1.5
                X[start_idx:end_idx, 10:] = np.random.randn(samples_per_class, 10) * 0.5 + 0.5
            # Other classes with intermediate patterns
            else:
                X[start_idx:end_idx] = np.random.randn(samples_per_class, n_features) * (0.5 + 0.3*class_idx)
            
            y[start_idx:end_idx] = class_idx
        
        # Shuffle
        perm = np.random.permutation(n_samples)
        X = X[perm]
        y = y[perm]
        
        return X, y
    
    def entropy_sampling(self, X_pool, model, pool_indices):
        """
        Entropy-based sampling for classification.
        Select samples where model is most uncertain about class probabilities.
        """
        entropies = []
        
        for idx in pool_indices:
            # Get class probabilities
            proba = model.predict_proba(X_pool[idx:idx+1])[0]
            
            # Compute entropy
            entropy = -np.sum(proba * np.log(proba + 1e-10))
            entropies.append(entropy)
        
        return np.array(entropies)
    
    def quantum_margin_sampling(self, X_pool, model, pool_indices):
        """
        Quantum-inspired margin sampling combined with uncertainty.
        Select samples near decision boundaries (small margin) with high entropy.
        """
        margins = []
        entropies = []
        
        for idx in pool_indices:
            proba = model.predict_proba(X_pool[idx:idx+1])[0]
            
            # Margin: difference between top two class probabilities
            sorted_proba = np.sort(proba)[::-1]
            margin = sorted_proba[0] - sorted_proba[1]
            margins.append(margin)
            
            # Entropy weight
            entropy = -np.sum(proba * np.log(proba + 1e-10))
            entropies.append(entropy)
        
        # Combine: prefer small margin (uncertain) AND high entropy
        margins = np.array(margins)
        entropies = np.array(entropies)
        
        # Normalize and combine
        normalized_margins = (margins - np.min(margins)) / (np.max(margins) - np.min(margins) + 1e-10)
        normalized_entropies = (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies) + 1e-10)
        
        combined_score = normalized_entropies * (1.0 - normalized_margins)
        return combined_score
    
    def run_experiment(self, n_iterations=8, batch_size=20, initial_pool_size=60):
        """Run discrete classification active learning experiment."""
        print("=" * 70)
        print("Discrete Materials Classification Experiment")
        print("=" * 70)
        
        # Generate dataset
        X_all, y_all = self.generate_synthetic_classification_dataset(n_samples=1200)
        
        # Normalize features
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_all)
        
        # Split into train pool and test set
        train_indices = np.random.choice(len(X_all), size=800, replace=False)
        test_indices = np.array([i for i in range(len(X_all)) if i not in train_indices])
        
        X_train, y_train = X_all[train_indices], y_all[train_indices]
        X_test, y_test = X_all[test_indices], y_all[test_indices]
        
        # Initialize labeled set
        labeled_idx = np.random.choice(len(X_train), size=initial_pool_size, replace=False)
        pool_idx = np.array([i for i in range(len(X_train)) if i not in labeled_idx])
        
        results = {
            'iteration': [],
            'accuracy_quantum': [],
            'accuracy_entropy': [],
            'accuracy_random': [],
            'f1_quantum': [],
            'f1_entropy': [],
            'f1_random': []
        }
        
        # Three parallel runs: quantum-margin, entropy, random
        for strategy_name, strategy_func in [
            ('quantum_margin', self.quantum_margin_sampling),
            ('entropy', self.entropy_sampling),
            ('random', lambda x, m, i: np.random.rand(len(i)))
        ]:
            print(f"\nTraining with {strategy_name.upper()} strategy...")
            
            labeled_idx = np.random.choice(len(X_train), size=initial_pool_size, replace=False)
            pool_idx = np.array([i for i in range(len(X_train)) if i not in labeled_idx])
            
            for iteration in range(n_iterations):
                # Train classifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train[labeled_idx], y_train[labeled_idx])
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                print(f"  Iter {iteration+1}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                
                # Store results
                if iteration == 0 and strategy_name == 'quantum_margin':
                    results['iteration'].append(iteration + 1)
                
                if strategy_name == 'quantum_margin':
                    results['accuracy_quantum'].append(accuracy)
                    results['f1_quantum'].append(f1)
                elif strategy_name == 'entropy':
                    results['accuracy_entropy'].append(accuracy)
                    results['f1_entropy'].append(f1)
                else:
                    results['accuracy_random'].append(accuracy)
                    results['f1_random'].append(f1)
                
                # Query next batch
                if len(pool_idx) > 0:
                    scores = strategy_func(X_train, model, pool_idx)
                    top_indices = np.argsort(scores)[-batch_size:]
                    selected_samples = pool_idx[top_indices]
                    
                    labeled_idx = np.concatenate([labeled_idx, selected_samples])
                    pool_idx = np.array([i for i in pool_idx if i not in selected_samples])
        
        # Fill in iteration list for all strategies
        if not results['iteration']:
            results['iteration'] = list(range(1, n_iterations + 1))
        
        return results
    
    def plot_results(self, results, save_path='classification_results.png'):
        """Visualize classification results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Discrete Classification: Active Learning Strategies', 
                     fontsize=14, fontweight='bold')
        
        iterations = results['iteration']
        
        # Accuracy comparison
        ax = axes[0]
        ax.plot(iterations, results['accuracy_quantum'], 'o-', label='Quantum-Margin', 
               linewidth=2, markersize=7, color='blue')
        ax.plot(iterations, results['accuracy_entropy'], 's--', label='Entropy Sampling', 
               linewidth=2, markersize=7, color='green')
        ax.plot(iterations, results['accuracy_random'], '^:', label='Random Sampling', 
               linewidth=2, markersize=7, color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Classification Accuracy')
        ax.set_title('Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 Score comparison
        ax = axes[1]
        ax.plot(iterations, results['f1_quantum'], 'o-', label='Quantum-Margin', 
               linewidth=2, markersize=7, color='blue')
        ax.plot(iterations, results['f1_entropy'], 's--', label='Entropy Sampling', 
               linewidth=2, markersize=7, color='green')
        ax.plot(iterations, results['f1_random'], '^:', label='Random Sampling', 
               linewidth=2, markersize=7, color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weighted F1 Score')
        ax.set_title('F1 Score Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    classifier = DiscreteClassificationFramework(n_classes=6)
    results = classifier.run_experiment(n_iterations=8, batch_size=20, initial_pool_size=60)
    
    # Save results
    with open('classification_results.json', 'w') as f:
        results_serializable = {
            'iteration': [int(x) for x in results['iteration']],
            'accuracy_quantum': [float(x) for x in results['accuracy_quantum']],
            'accuracy_entropy': [float(x) for x in results['accuracy_entropy']],
            'accuracy_random': [float(x) for x in results['accuracy_random']],
            'f1_quantum': [float(x) for x in results['f1_quantum']],
            'f1_entropy': [float(x) for x in results['f1_entropy']],
            'f1_random': [float(x) for x in results['f1_random']]
        }
        json.dump(results_serializable, f, indent=2)
    
    print("\nResults saved to classification_results.json")
    
    # Plot results
    classifier.plot_results(results)
