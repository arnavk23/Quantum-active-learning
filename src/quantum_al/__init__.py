"""quantum_al: a covariance-aware, quantum-inspired uncertainty formalism
for active learning, benchmarked honestly against classical baselines on
real Materials Project data, with a verified quantum circuit realization.

Key entry points:
    quantum_al.operator.QuantumObservableBank   -- the core formalism (Eq. 1-6)
    quantum_al.operator_v3.TreeEnsembleQuantumSelector -- the residual-coupled fix
    quantum_al.circuit                          -- real Qiskit circuit realization
    quantum_al.baselines                        -- 9 classical AL baselines
    quantum_al.data_utils.load_task             -- load real Materials Project data
    quantum_al.fetch_data                       -- (re)fetch data from the MP API
"""

__version__ = "0.1.0"
