# quantum_al: Correlation-Aware Active Learning for Materials Discovery

[![Tests](https://github.com/arnavk23/Quantum-active-learning/actions/workflows/tests.yml/badge.svg)](https://github.com/arnavk23/Quantum-active-learning/actions/workflows/tests.yml)

A Python package implementing and honestly benchmarking two attempts at a
correlation-aware acquisition function for active learning in materials
discovery: a joint expected-information-gain score derived from Bayesian
experimental design (the main result), and an earlier covariance-aware
quantum-inspired formalism. Both are tested against classical baselines on
real Materials Project data. See [`paper.md`](paper.md) for the software
description (JOSS format) and [`papers/`](papers/) for the full research
manuscript.

## Honest summary of findings

**Main result: joint expected information gain.** For $K$ real, jointly-labeled
target properties, `src/quantum_al/joint_eig.py` scores a candidate by
$\mathrm{JEIG}(x) = \tfrac12\log\det(\Sigma_{\mathrm{pred}}(x) + R)$, the
ensemble-disagreement estimate of joint expected information gain. We prove
(and verify numerically to $10^{-8}$) that summing the per-task marginal terms
and subtracting this joint term always equals the total correlation among the
$K$ predictive uncertainties, a quantity that is exactly zero when the tasks
are uncorrelated, in which case the score collapses exactly to ordinary
per-task ensemble-variance scoring. Tested against that classical-limit
ablation and against random sampling on four real, independently-measured
Materials Project property pairs (spanning label correlation $r=-0.37$ to
$0.33$, $n=49$ to $498$), the joint score **never significantly beats the
correlation-blind ablation in any of the four pairs**, and beats random
sampling significantly in only one (the largest, most strongly correlated
pair). The total-correlation term is measurably non-zero in every pair, so
the theorem is not vacuous on this data, but it does not translate into a
reliable accuracy gain.

**Earlier attempt: quantum-inspired covariance.** A correctly-implemented,
unit-tested covariance-aware quantum-inspired formalism
(`src/quantum_al/operator.py`) was evaluated against 9 standard
active-learning baselines on 5 real Materials Project regression tasks. As
originally specified, it loses on 4 of 5 tasks, no paired comparison survives
Holm-Bonferroni correction, and an ablation shows its covariance term has an
effect indistinguishable from noise (+0.07% R², vs. trial-to-trial σ of
5-8%). Diagnosing the cause (the score never sees the downstream model's own
residuals) and coupling the state encoding to a random forest's per-tree
disagreement brings it to statistical parity with the best baseline on every
task, but a further ablation shows the gain comes entirely from the
disagreement signal, not the quantum-inspired machinery. A real quantum
circuit realization (Qiskit) confirms the formalism matches its classical
simulation exactly and characterizes its NISQ cost: hundreds of Pauli
measurement terms per quantity, cut 4-12x by standard measurement grouping
and a further 2-3x by building observables sparse from the start.

**Across two structurally unrelated attempts, one an analogy and one a
derivation, correlation-aware aggregation of multiple uncertainty sources
does not reliably outperform directly measuring ensemble disagreement.**

Full results, diagnosis, and every real number behind the tables and figures:
[`results/SUMMARY.md`](results/SUMMARY.md).

An earlier version of this repository/paper claimed a 35% sample-efficiency
improvement and p<0.01 significance over 9 baselines, and a formalism that
worked as originally specified. Those numbers did not reproduce from any code
in this repository when actually run and have been retracted; everything
above is what the real, rerun experiments show.

## Installation

```bash
git clone https://github.com/arnavk23/Quantum-active-learning.git
cd Quantum-active-learning
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -e ".[test,circuit]"
```

The `circuit` extra installs Qiskit and Qiskit Aer, needed only for
`src/quantum_al/circuit.py` and the quantum-hardware-realization benchmark.
The `test` extra installs pytest.

## Verify the install

```bash
pytest tests/ -v
```

This runs the correctness checks referenced throughout the papers: the
classical-limit reduction proof (`tests/test_operator.py`), the joint-EIG
total-correlation decomposition proof (`tests/test_joint_eig.py`), the
circuit-vs-classical exact-match check (`tests/test_circuit.py`, skipped if
Qiskit is not installed), and smoke tests for all 9 baselines
(`tests/test_baselines.py`).

## Repository structure

```
src/quantum_al/       the installable package
  joint_eig.py            main result: joint expected-information-gain score + its theorem
  operator.py             earlier attempt: covariance-aware quantum-inspired formalism (Eq. 1-6)
  operator_v2.py           two failed narrow fix attempts (domain grouping, importance weighting)
  operator_v3.py           the residual-coupled fix that reaches parity
  operator_sparse.py       sparse-by-construction observables for the quantum formalism
  circuit.py               real Qiskit circuit realization + NISQ resource tools
  baselines.py             9 classical active-learning acquisition strategies
  data_utils.py            load real Materials Project data, incl. multi-property inner joins
  fetch_data.py            (re)fetch data from the Materials Project API

benchmarks/            runnable scripts that produced every table/figure
  run_joint_eig_experiment.py    main result: 4-pair real multi-property test
  run_primary_benchmark.py       quantum-formalism primary comparison + significance
  run_improvement_attempt.py     the two failed narrow fixes
  run_v3_test.py / run_v3_ablation.py / run_v3_all_tasks.py   the residual-coupled fix + its ablation
  run_quantum_circuit_experiment.py   NISQ feasibility characterization
  run_sparse_observable_experiment.py   sparse-vs-dense observable comparison
  make_paper_figures.py          regenerates figures/*.pdf from results/*.json

tests/                 pytest suite, run in CI on every push
papers/                 full research manuscript (npj-Computational-Materials-style)
                        plus an IEEE-conference-style draft, superseded prior
                        drafts, and reviewer feedback, kept for provenance
figures/                figures embedded in the papers, generated from results/ (tracked in git)
results/                raw JSON output backing every number in the papers (tracked in git; regenerate via benchmarks/)
data/                   real Materials Project data (gitignored; regenerate via src/quantum_al/fetch_data.py)
```

## Regenerating the data and results

```bash
export MP_API_KEY=your_materials_project_api_key   # https://next-gen.materialsproject.org/api
python -m quantum_al.fetch_data
python benchmarks/run_joint_eig_experiment.py --tasks band_gap formation_energy --n-trials 5 --out joint_eig_experiment.json
python benchmarks/run_joint_eig_experiment.py --tasks formation_energy magnetic_moment --n-trials 5 --out joint_eig_experiment_replication.json
python benchmarks/run_joint_eig_experiment.py --tasks band_gap magnetic_moment --n-trials 5 --out joint_eig_experiment_pair_bg_mm.json
python benchmarks/run_joint_eig_experiment.py --tasks bulk_modulus dielectric_constant --n-trials 5 --n0 15 --t-iters 5 --batch-size 3 --out joint_eig_experiment_pair_bm_dc.json
python benchmarks/run_primary_benchmark.py --stage all
python benchmarks/run_v3_all_tasks.py
python benchmarks/run_quantum_circuit_experiment.py   # needs the [circuit] extra
python benchmarks/make_paper_figures.py
```

## Citation

See [`CITATION.cff`](CITATION.cff), or cite the accompanying manuscript once
published (see [`papers/`](papers/) for current drafts).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Contact

Arnav Kapoor — arnavkapoor23@iiserb.ac.in
