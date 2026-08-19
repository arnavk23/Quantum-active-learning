---
title: 'quantum_al: benchmarking quantum-inspired active learning for materials discovery'
tags:
  - Python
  - active learning
  - materials discovery
  - materials informatics
  - quantum computing
  - quantum-inspired algorithms
  - uncertainty quantification
authors:
  - name: Arnav Kapoor
    orcid: 0009-0007-9818-7908
    affiliation: 1
affiliations:
  - name: Indian Institute of Science Education and Research Bhopal, India
    index: 1
date: 19 August 2026
bibliography: paper.bib
---

# Summary

`quantum_al` is a Python package for building and rigorously testing quantum-inspired
acquisition functions for active learning in materials discovery. It provides three
things together, which we could not find combined in any existing open-source tool:
(1) a correct, unit-tested implementation of a covariance-aware, quantum-inspired
uncertainty formalism, state encoding into a feature Hilbert space, non-commuting
Hermitian observables, and complex-coefficient covariance aggregation, with a proven
and numerically verified classical-limit reduction; (2) nine working classical
active-learning baselines (uncertainty sampling, Query-by-Committee, Expected
Improvement, BADGE, CoreSet, and others) alongside a real-data benchmark harness
that pulls genuine DFT-computed properties directly from the Materials Project API
[@jain2013materials], runs paired significance testing with Holm-Bonferroni
correction, and reports results with no synthetic stand-ins; and (3) an actual
quantum circuit realization of the formalism, built with Qiskit [@qiskit2024], that
is verified to reproduce the classical formula to floating-point precision and comes
with tooling to characterize its near-term hardware cost: Pauli measurement
overhead, shot-count convergence of the resulting acquisition ranking, sensitivity
to gate noise, and the effect of measurement grouping.

# Statement of need

Active learning is a standard tool for reducing the labeling cost of materials
discovery pipelines, where each new label can mean a DFT calculation or a synthesis
experiment [@lookman2019active; @vandermause2020fly; @pyzerknapp2022accelerating].
A growing body of work proposes quantum and quantum-inspired uncertainty measures
for this setting [@biamonte2017quantum; @schuld2021machine], motivated by the idea
that non-commuting observables and complex coupling coefficients can represent
cross-property correlations that a scalar acquisition score discards. What is
missing is tooling to test such proposals honestly: against strong classical
baselines, on real materials data rather than synthetic proxies, with proper paired
statistics, and, where a "quantum-inspired" formalism claims a connection to actual
quantum computation, against a real circuit rather than only its classical
simulation.

`quantum_al` was built to fill that gap after an earlier, less rigorous evaluation
of exactly this kind of formalism produced numbers that did not reproduce when
actually run. The package is designed so that every claim it can produce is
checked: the operator formalism includes a numerical proof of its own classical
limit, the benchmark harness never falls back to simulated or illustrative numbers
on failure, and the circuit module asserts exact agreement with the classical
formula before any circuit-derived resource estimate is trusted. `quantum_al` is
useful to two audiences: materials-informatics researchers who want a reproducible,
real-data harness for comparing acquisition strategies, and quantum-computing
researchers who want a concrete, worked example of translating a quantum-inspired
machine-learning construction into an actual circuit and characterizing its
near-term feasibility. The package has already been used to produce a full
empirical study of one such formalism, reported in an accompanying manuscript
[@kapoor2026].

# Functionality

The core package lives under `src/quantum_al/`:

- `operator.py` implements the covariance-aware formalism (state encoding, Hermitian
  observables, variance, symmetrized covariance, complex-coefficient acquisition
  score) and includes `self_test()`, verifying non-commutativity and the exact
  classical-limit reduction.
- `operator_v3.py` implements a residual-coupled variant that encodes state from a
  downstream model's per-tree predictions instead of raw features.
- `circuit.py` implements the real Qiskit circuit realization: amplitude encoding,
  Pauli decomposition, exact and shot-based expectation estimation, depolarizing
  noise injection, and measurement-grouping utilities.
- `baselines.py` implements nine classical active-learning acquisition strategies
  behind a common `select_next_experiments(X_candidates, X_train, y_train,
  n_select)` interface.
- `data_utils.py` and `fetch_data.py` load and (re)fetch real Materials Project
  regression and classification data.

A minimal usage example:

```python
from quantum_al.operator import QuantumObservableBank, default_feature_groups
from quantum_al.baselines import get_all_baselines

d = 21  # feature dimensionality
bank = QuantumObservableBank(d, default_feature_groups(d))
scores = bank.batch_scores(X_labeled, X_candidates)  # acquisition scores

baselines = get_all_baselines()  # 9 classical strategies, same interface
```

The `benchmarks/` directory contains the runnable experiment scripts (primary
benchmark, ablations, the residual-coupled variant and its own ablation, and the
quantum circuit resource characterization) used to produce every table and figure
in the accompanying manuscript, and `tests/` contains the automated correctness
tests referenced above, run on every push via GitHub Actions.

# Acknowledgements

This work uses data from the Materials Project [@jain2013materials] and the Qiskit
SDK [@qiskit2024].

# References
