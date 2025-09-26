"""
Quantum Computing for Materials Science: Advanced Demo

This script demonstrates:
- Quantum circuit basics (gates, measurement, visualization)
- Superposition and entanglement
- Simulation and state analysis
- Quantum algorithms: Grover's algorithm, VQE, QPE
- Hybrid quantum-classical ML integration
- Error analysis and advanced visualization

Each section includes code, comments, and explanations for educational and reproducibility purposes.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from qiskit import QuantumCircuit, transpile
try:
    from qiskit.providers.aer import AerSimulator
except ImportError:
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        AerSimulator = None
from qiskit.visualization import plot_histogram
try:
    from qiskit.circuit.library import GroverOperator
    from qiskit.algorithms import Grover, AmplificationProblem
    grover_available = True
except ImportError:
    grover_available = False

# 1. Create and visualize a simple quantum circuit
def demo_circuit():
    # Ensure results directory exists
    os.makedirs('./results', exist_ok=True)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    try:
        fig = qc.draw('mpl')
        # Only save if it's a matplotlib Figure
        import matplotlib.figure
        if isinstance(fig, matplotlib.figure.Figure):
            fig.savefig('./results/quantum_circuit.png')
            plt.close(fig)
        else:
            # Fallback: save text drawing
            with open('./results/quantum_circuit.txt', 'w') as f:
                f.write(str(qc.draw()))
            print("Saved quantum circuit as text drawing.")
    except Exception as e:
        print(f"Warning: Could not generate mpl circuit figure. Error: {e}")
        with open('./results/quantum_circuit.txt', 'w') as f:
            f.write(str(qc.draw()))
    return qc

# 2. Simulate the quantum circuit
def simulate_circuit(qc):
    if AerSimulator is None:
        print("AerSimulator is not available. Skipping quantum simulation.")
        return None
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=1024).result()
    counts = result.get_counts()
    plot_histogram(counts)
    plt.title('Quantum Circuit Measurement Results')
    plt.savefig('./results/quantum_circuit_hist.png')
    plt.close()
    return counts

# 3. Grover's algorithm demo
def grover_demo():
    if not grover_available or AerSimulator is None:
        print("Grover's algorithm modules or AerSimulator not available in this Qiskit version.")
        return None
    try:
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        problem = AmplificationProblem(oracle=grover_op)
        grover = Grover()
        simulator = AerSimulator()
        result = grover.amplify(problem, quantum_instance=simulator)
        print("Grover's algorithm result:", result.top_measurement)
        return result.top_measurement
    except Exception as e:
        print(f"Grover's algorithm failed: {e}")
        return None


def quantum_circuit_basics():
    """
    Demonstrate single-qubit gates, measurement, and state vector simulation.
    """
    print("\n--- Quantum Circuit Basics ---")
    # Create a single-qubit circuit
    qc = QuantumCircuit(1)
    qc.h(0)  # Hadamard gate: creates superposition
    qc.x(0)  # Pauli-X gate: flips qubit
    qc.measure_all()
    # Visualize circuit
    try:
        fig = qc.draw('mpl')
        import matplotlib.figure
        if isinstance(fig, matplotlib.figure.Figure):
            fig.savefig('./results/single_qubit_circuit.png')
            plt.close(fig)
        else:
            with open('./results/single_qubit_circuit.txt', 'w') as f:
                f.write(str(qc.draw()))
    except Exception as e:
        print(f"Warning: Could not generate mpl figure. Error: {e}")
        with open('./results/single_qubit_circuit.txt', 'w') as f:
            f.write(str(qc.draw()))
    # Simulate state vector
    try:
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()
        print("Statevector probabilities:", probs)
        # Bar plot of probabilities
        plt.figure()
        plt.bar(list(probs.keys()), list(probs.values()))
        plt.title('Single Qubit State Probabilities')
        plt.xlabel('State')
        plt.ylabel('Probability')
        plt.savefig('./results/single_qubit_probs.png')
        plt.close()
    except Exception as e:
        print(f"Statevector simulation failed: {e}")

def superposition_demo():
    """
    Demonstrate quantum superposition using Hadamard gate.
    """
    print("\n--- Superposition Demo ---")
    qc = QuantumCircuit(1)
    qc.h(0)
    # No measurement for statevector simulation
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()
    print("Superposition statevector probabilities:", probs)
    plt.figure()
    plt.bar(list(probs.keys()), list(probs.values()))
    plt.title('Superposition State Probabilities (Hadamard)')
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.savefig('./results/superposition_probs.png')
    plt.close()

def entanglement_demo():
    """
    Demonstrate quantum entanglement using Bell state circuit.
    """
    print("\n--- Entanglement Demo (Bell State) ---")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()
    print("Bell statevector probabilities:", probs)
    plt.figure()
    plt.bar(list(probs.keys()), list(probs.values()))
    plt.title('Bell State Probabilities (Entanglement)')
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.savefig('./results/bell_state_probs.png')
    plt.close()

def grover_demo_advanced():
    """
    Demonstrate Grover's algorithm for quantum search (if available).
    """
    print("\n--- Grover's Algorithm Demo ---")
    try:
        from qiskit.circuit.library import GroverOperator
        from qiskit.algorithms import Grover, AmplificationProblem
        from qiskit.providers.aer import AerSimulator
    except ImportError:
        print("Grover's algorithm modules not available in this Qiskit version.")
        return None
    try:
        # Oracle for |11> state
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        problem = AmplificationProblem(oracle=grover_op)
        grover = Grover()
        simulator = AerSimulator()
        result = grover.amplify(problem, quantum_instance=simulator)
        print("Grover's algorithm result (top measurement):", result.top_measurement)
        # Save result to file
        with open('./results/grover_result.txt', 'w') as f:
            f.write(f"Grover's algorithm top measurement: {result.top_measurement}\n")
        return result.top_measurement
    except Exception as e:
        print(f"Grover's algorithm failed: {e}")
        return None

def vqe_demo():
    """
    Demonstrate Variational Quantum Eigensolver (VQE) for ground state energy estimation (stub, robust to missing modules).
    """
    print("\n--- VQE Demo ---")
    try:
        from qiskit.algorithms import VQE
        from qiskit.circuit.library import TwoLocal
        from qiskit.opflow import Z, I
        from qiskit.providers.aer import AerSimulator
        from qiskit.algorithms.optimizers import COBYLA
    except ImportError:
        print("VQE modules not available in this Qiskit version.")
        return None
    try:
        # Simple Hamiltonian: ZI + IZ
        hamiltonian = Z ^ I + I ^ Z
        ansatz = TwoLocal(2, ['ry', 'rz'], 'cz')
        optimizer = COBYLA(maxiter=100)
        vqe = VQE(ansatz, optimizer=optimizer)
        simulator = AerSimulator()
        result = vqe.compute_minimum_eigenvalue(hamiltonian, quantum_instance=simulator)
        print("VQE ground state energy:", result.eigenvalue)
        with open('./results/vqe_result.txt', 'w') as f:
            f.write(f"VQE ground state energy: {result.eigenvalue}\n")
        return result.eigenvalue
    except Exception as e:
        print(f"VQE failed: {e}")
        return None

def qpe_demo():
    """
    Demonstrate Quantum Phase Estimation (QPE) for phase estimation (stub, robust to missing modules).
    """
    print("\n--- QPE Demo ---")
    try:
        from qiskit.algorithms import PhaseEstimation
        from qiskit.circuit.library import QFT
        from qiskit.providers.aer import AerSimulator
    except ImportError:
        print("QPE modules not available in this Qiskit version.")
        return None
    try:
        # Example: Estimate phase of controlled-Z gate
        unitary = QuantumCircuit(1)
        unitary.rz(np.pi/4, 0)
        qpe = PhaseEstimation(num_evaluation_qubits=3)
        simulator = AerSimulator()
        result = qpe.estimate(unitary, np.exp(1j * np.pi/4), quantum_instance=simulator)
        print("QPE estimated phase:", result)
        with open('./results/qpe_result.txt', 'w') as f:
            f.write(f"QPE estimated phase: {result}\n")
        return result
    except Exception as e:
        print(f"QPE failed: {e}")
        return None

def hybrid_ml_demo():
    """
    Demonstrate hybrid quantum-classical ML: quantum feature extraction for classical ML.
    """
    print("\n--- Hybrid Quantum-Classical ML Demo ---")
    # This demo shows how quantum circuits can be used to extract features for classical ML models.
    # Quantum feature: expectation value of PauliZ after encoding classical data into a quantum state.
    # Classical feature: raw input value.
    # We compare accuracy and confusion matrices for both approaches on noisy synthetic data.
    # References:
    # - Schuld, M., Sinayskiy, I., & Petruccione, F. (2015). An introduction to quantum machine learning. Contemporary Physics, 56(2), 172-185.
    # - Qiskit Textbook: https://qiskit.org/textbook/ch-machine-learning/
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        # Generate noisy synthetic data
        np.random.seed(42)
        X = np.random.rand(200, 1)
        y = (X[:, 0] > 0.5).astype(int)
        # Add noise to labels
        noise = np.random.binomial(1, 0.1, size=y.shape)
        y_noisy = np.abs(y - noise)  # flip 10% of labels

        # Quantum feature extraction: encode data into quantum circuit and extract expectation value
        def quantum_feature(x):
            # Encode classical value x into a single-qubit quantum state using Ry rotation
            qc = QuantumCircuit(1)
            qc.ry(x * np.pi, 0)
            from qiskit.quantum_info import Statevector, Pauli
            sv = Statevector.from_instruction(qc)
            # Extract expectation value of PauliZ operator (quantum feature)
            z_exp = sv.expectation_value(Pauli('Z'))
            return np.real(z_exp)
        X_quantum = np.array([quantum_feature(x[0]) for x in X]).reshape(-1, 1)

        # Classical feature (just X)
        # This is the raw input, used for comparison with quantum feature
        X_classical = X

        # Train/test split
        # Quantum feature split
        Xq_train, Xq_test, y_train, y_test = train_test_split(X_quantum, y_noisy, test_size=0.2, random_state=42)
        # Classical feature split (same labels)
        Xc_train, Xc_test, _, _ = train_test_split(X_classical, y_noisy, test_size=0.2, random_state=42)

        # Train logistic regression on quantum features
        clf_q = LogisticRegression().fit(Xq_train, y_train)
        yq_pred = clf_q.predict(Xq_test)
        acc_q = accuracy_score(y_test, yq_pred)

        # Train logistic regression on classical features
        clf_c = LogisticRegression().fit(Xc_train, y_train)
        yc_pred = clf_c.predict(Xc_test)
        acc_c = accuracy_score(y_test, yc_pred)

        print(f"Hybrid ML accuracy (quantum feature): {acc_q:.3f}")
        print(f"Classical ML accuracy (classical feature): {acc_c:.3f}")

        # Visualization: feature scatter
        # Compare quantum and classical feature distributions
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.scatter(X_quantum, y_noisy, c=y_noisy, cmap='coolwarm', alpha=0.7)
        plt.title('Quantum Feature vs. Label')
        plt.xlabel('Quantum Feature (Z expectation)')
        plt.ylabel('Label')
        plt.subplot(1,2,2)
        plt.scatter(X_classical, y_noisy, c=y_noisy, cmap='coolwarm', alpha=0.7)
        plt.title('Classical Feature vs. Label')
        plt.xlabel('Classical Feature (X)')
        plt.ylabel('Label')
        plt.tight_layout()
        plt.savefig('./results/quantum_vs_classical_feature_scatter.png')
        plt.close()

        # Error analysis: confusion matrices
        # Show confusion matrices for quantum and classical classifiers
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm_q = confusion_matrix(y_test, yq_pred)
        cm_c = confusion_matrix(y_test, yc_pred)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        ConfusionMatrixDisplay(cm_q).plot(ax=plt.gca(), cmap='Blues')
        plt.title('Quantum Feature Confusion Matrix')
        plt.subplot(1,2,2)
        ConfusionMatrixDisplay(cm_c).plot(ax=plt.gca(), cmap='Blues')
        plt.title('Classical Feature Confusion Matrix')
        plt.tight_layout()
        plt.savefig('./results/quantum_vs_classical_confusion_matrix.png')
        plt.close()

        # Measurement distribution for quantum feature
        # Show how quantum feature values are distributed
        plt.figure()
        plt.hist(X_quantum, bins=20, color='purple', alpha=0.7)
        plt.title('Distribution of Quantum Feature (Z expectation)')
        plt.xlabel('Quantum Feature Value')
        plt.ylabel('Count')
        plt.savefig('./results/quantum_feature_distribution.png')
        plt.close()

    # Add comments for reproducibility and educational value
    # - Quantum feature is expectation value of PauliZ after encoding classical data into quantum state
    # - Classical feature is raw input
    # - Compare accuracy and confusion matrices
    # - Visualize feature distributions
    except Exception as e:
        print(f"Hybrid ML demo failed: {e}")
    print(f"Classical ML accuracy (classical feature): {acc_c:.3f}")
    # Accuracy may be similar for this simple synthetic dataset, but quantum features can be useful for more complex data.

    # Visualization: feature scatter
    # Compare quantum and classical feature distributions
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(X_quantum, y_noisy, c=y_noisy, cmap='coolwarm', alpha=0.7)
    plt.title('Quantum Feature vs. Label')
    plt.xlabel('Quantum Feature (Z expectation)')
    plt.ylabel('Label')
    plt.subplot(1,2,2)
    plt.scatter(X_classical, y_noisy, c=y_noisy, cmap='coolwarm', alpha=0.7)
    plt.title('Classical Feature vs. Label')
    plt.xlabel('Classical Feature (X)')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig('./results/quantum_vs_classical_feature_scatter.png')
    plt.close()

    # Error analysis: confusion matrices
    # Show confusion matrices for quantum and classical classifiers
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm_q = confusion_matrix(y_test, yq_pred)
    cm_c = confusion_matrix(y_test, yc_pred)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    ConfusionMatrixDisplay(cm_q).plot(ax=plt.gca(), cmap='Blues')
    plt.title('Quantum Feature Confusion Matrix')
    plt.subplot(1,2,2)
    ConfusionMatrixDisplay(cm_c).plot(ax=plt.gca(), cmap='Blues')
    plt.title('Classical Feature Confusion Matrix')
    plt.tight_layout()
    plt.savefig('./results/quantum_vs_classical_confusion_matrix.png')
    plt.close()

    # Measurement distribution for quantum feature
    # Show how quantum feature values are distributed
    plt.figure()
    plt.hist(X_quantum, bins=20, color='purple', alpha=0.7)
    plt.title('Distribution of Quantum Feature (Z expectation)')
    plt.xlabel('Quantum Feature Value')
    plt.ylabel('Count')
    plt.savefig('./results/quantum_feature_distribution.png')
    plt.close()
  
if __name__ == "__main__":
    print("Quantum Computing Demo for Surrogate Modeling (Advanced)")
    quantum_circuit_basics()
    superposition_demo()
    entanglement_demo()
    qc = demo_circuit()
    counts = simulate_circuit(qc)
    print("Quantum circuit measurement counts:", counts)
    grover_result = grover_demo_advanced()
    print("Grover's algorithm top measurement:", grover_result)
    vqe_result = vqe_demo()
    print("VQE ground state energy:", vqe_result)
    qpe_result = qpe_demo()
    print("QPE estimated phase:", qpe_result)
    hybrid_ml_demo()
    print("Saved quantum circuit and results visualizations to ./results/")
