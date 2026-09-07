---
title: 'quantum_al: a reproducible harness for benchmarking correlation-aware active-learning acquisition functions on real materials data'
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
date: 8 September 2026
bibliography: paper.bib
---

# Summary

`quantum_al` is a Python package for building and rigorously testing acquisition
functions that try to exploit correlation between multiple materials properties
during active learning, the process of sequentially choosing which candidate
materials to label next under a fixed labeling budget. The package provides three
things together: (1) two independently-implemented, unit-tested acquisition-score
constructions, a joint expected-information-gain score derived from Bayesian
experimental design with a proved decomposition theorem, and a covariance-aware
quantum-inspired formalism that encodes candidates as states in a Hilbert space,
each verified against its own closed-form mathematical reduction; (2) a real-data
benchmark harness that pulls genuine DFT-computed properties directly from the
Materials Project API [@jain2013materials], runs nine classical active-learning
baselines behind a common interface, and reports paired significance tests with
Holm-Bonferroni correction, never falling back to synthetic or illustrative
numbers; and (3) an actual quantum circuit realization of the Hilbert-space
formalism, built with Qiskit [@qiskit2024], verified to reproduce its classical
simulation to floating-point precision, with tooling to characterize near-term
hardware cost.

# Statement of need

Active learning is a standard tool for reducing the labeling cost of materials
discovery pipelines, where each new label can mean a DFT calculation or a
synthesis experiment [@lookman2019active; @vandermause2020fly;
@pyzerknapp2022accelerating]. Materials properties are often correlated, since
they share the same underlying composition and structure, and a growing body of
work proposes ways to exploit that correlation, including quantum and
quantum-inspired uncertainty measures [@biamonte2017quantum; @schuld2021machine],
motivated by the idea that non-commuting observables or joint information-theoretic
scores can represent cross-property structure that a scalar acquisition score
discards. What has been missing is tooling to test such proposals honestly:
against strong classical baselines, on real materials data rather than synthetic
proxies, with proper paired statistics across multiple independent property
pairs, and, where a construction claims a connection to actual quantum
computation, against a real circuit rather than only its classical simulation.
`quantum_al` targets two audiences: materials-informatics researchers who want a
reproducible, real-data harness for comparing acquisition strategies (including
ones they design themselves, behind the package's shared selector interface),
and quantum-computing researchers who want a worked example of translating a
quantum-inspired construction into an actual circuit and characterizing its
near-term feasibility.

# State of the field

General-purpose active-learning libraries such as modAL [@danka2018modal] and
ALiPy [@tang2019alipy] provide broad collections of classical acquisition
strategies behind a scikit-learn-compatible interface, but neither includes
materials-specific data loaders, a real Materials Project benchmark protocol, or
any mechanism for representing cross-property correlation in the acquisition
score. Conversely, the quantum-inspired and quantum-machine-learning literature
[@biamonte2017quantum; @schuld2021machine] proposes operator-based and
information-theoretic constructions of this kind, but rarely ships as tested,
runnable software, and essentially never includes a real quantum circuit
realization verified against the construction's own classical closed form.
`quantum_al` fills the gap between these two categories: it is a materials-data
active-learning benchmark, in the spirit of modAL and ALiPy, purpose-built to
host and honestly evaluate correlation-aware acquisition constructions,
including a real Qiskit circuit realization of one such construction. We did not
find an existing package that combines a real-data statistical benchmark, a
provably-grounded information-theoretic acquisition score with a checked
decomposition theorem, and a verified quantum circuit realization in one tested
codebase, which is the specific gap this package fills rather than reimplementing
general-purpose active-learning infrastructure that modAL and ALiPy already
provide well.

# Software design

The package is organized so that every acquisition-score construction is
self-contained, unit-tested against its own mathematical properties, and
callable behind the same `select_next_experiments(X_candidates, X_train,
y_train, n_select)` interface as the nine classical baselines, so that adding a
new construction or a new baseline does not require touching the benchmark
harness. `src/quantum_al/joint_eig.py` implements the joint expected-information-gain
score and its classical-limit ablation, with a `self_test()` that checks the
Hadamard-inequality decomposition identity numerically; `src/quantum_al/operator.py`
implements the Hilbert-space formalism with an analogous `self_test()` for its
own classical-limit reduction; `src/quantum_al/circuit.py` implements the Qiskit
circuit realization and asserts exact agreement with the classical formula
before any circuit-derived resource estimate is used. A deliberate design choice
runs through all of this: the benchmark harness in `benchmarks/` never
substitutes simulated or illustrative numbers when a real computation fails, and
raises rather than silently degrading, a direct response to an earlier,
less rigorous evaluation of this kind of construction that produced numbers
which did not reproduce when actually run. `data_utils.py` centralizes loading
of real Materials Project data, including an inner-join utility for
constructing genuinely correlated multi-property datasets from separately-fetched
per-property files, so that new correlation-aware constructions can be tested
against real, non-synthetic multi-property data without writing new data
plumbing. The trade-off of this design is that every acquisition strategy must
conform to a fairly narrow interface (a scored ranking over a candidate batch,
given a labeled pool); strategies that need a fundamentally different querying
protocol, such as multi-fidelity or cost-aware active learning, are out of scope
for the current interface and would need it extended.

A minimal usage example:

```python
from quantum_al.joint_eig import JointEIGSelector
from quantum_al.data_utils import load_multi_task, standardize
from quantum_al.baselines import get_all_baselines

X, Y, meta = load_multi_task(["band_gap", "formation_energy"])
X = standardize(X)
selector = JointEIGSelector(n_estimators=200, seed=0)
idx, scores, info = selector.select_next_experiments(
    X_candidates, X_train, Y_train, n_select=15
)

baselines = get_all_baselines()  # 9 classical strategies, same interface
```

The `benchmarks/` directory contains the runnable experiment scripts (the
multi-property joint-acquisition test, the primary single-property benchmark,
ablations, and the quantum circuit resource characterization) used to produce
every table and figure in the accompanying manuscripts, and `tests/` contains
the automated correctness tests referenced above, run on every push via GitHub
Actions.

# Research impact statement

`quantum_al` has been used by its author to conduct two full empirical studies
of correlation-aware active-learning acquisition, reported in an accompanying
manuscript [@kapoor2026]: a joint expected-information-gain score tested across
four real, independently-measured Materials Project property pairs, and a
covariance-aware quantum-inspired formalism tested across nine baselines and
five real regression tasks, including a real quantum circuit realization and
near-term hardware cost characterization. Both studies rely on the package's
self-tests and real-data-only benchmark policy for their validity, and the
manuscript's honest, partly-negative findings are a direct product of tooling
that would not let a failed computation be silently replaced with an
illustrative number. The software has not yet been adopted by research groups outside the author's
own work; the manuscript is currently in preparation and not yet publicly
posted. Its most credible near-term significance is re-use potential rather
than existing external adoption: the shared selector interface, the real
Materials Project data pipeline, and the self-testing discipline are designed
specifically so a third party can test a new correlation-aware construction of
their own against the same real baselines and data, without rebuilding this
infrastructure.

# AI usage disclosure

Generative AI (Claude, Anthropic, multiple model versions across development
sessions from 2025 to 2026) was used extensively in this project: for writing
and refactoring the majority of the Python implementation in `src/quantum_al/`
and `benchmarks/`, for drafting the automated test suite in `tests/`, and for
drafting this paper and the accompanying research manuscripts. AI assistance
also included fact-checking an earlier version of this project's own claims by
directly executing its code, which is what surfaced that an earlier evaluation
had produced numbers that did not reproduce. The human author directed this
work throughout: framing the research questions, choosing which experimental
directions and fixes to pursue, setting the requirement that no result be
reported without being produced by an actual, passing run of real code on real
data, rejecting AI-generated drafts and results that did not meet this bar, and
reviewing and validating the final code, tests, statistical methodology, and
reported findings before they were used. The author takes full responsibility
for the correctness, originality, and reporting of everything in this
repository and the accompanying manuscripts.

# Acknowledgements

This work uses data from the Materials Project [@jain2013materials] and the
Qiskit SDK [@qiskit2024].

# References
