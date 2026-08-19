# quantum_al: Quantum-Inspired Active Learning for Materials Discovery

[![Tests](https://github.com/arnavk23/Quantum-active-learning/actions/workflows/tests.yml/badge.svg)](https://github.com/arnavk23/Quantum-active-learning/actions/workflows/tests.yml)

A Python package implementing a covariance-aware, quantum-inspired uncertainty
formalism for active learning in materials discovery, benchmarked honestly
against classical baselines on real Materials Project data, with a verified
quantum circuit realization. See [`paper.md`](paper.md) for the software
description (JOSS format) and [`papers/`](papers/) for the full research
manuscripts.

## Honest summary of findings

A correctly-implemented, unit-tested version of the covariance-aware formalism
(`src/quantum_al/operator.py`) was evaluated against 9 standard active-learning
baselines on 5 real Materials Project regression tasks (band gap, formation
energy, bulk modulus, magnetic moment, dielectric constant; ~1000 materials
each). **As originally specified, the method does not outperform the
baselines**: it loses on 4 of 5 tasks, no paired comparison survives
Holm-Bonferroni correction, and an ablation study shows covariance coupling,
the mechanism the method is built around, has an effect indistinguishable
from noise (+0.07% R², vs. trial-to-trial σ of 5-8%).

Diagnosing the cause (the acquisition score never sees the downstream model's
own residuals) and fixing it directly, by coupling the state encoding to a
random forest's per-tree disagreement, brings the method to statistical
parity with the best baseline on every task. A further ablation shows that
parity comes entirely from the disagreement signal, not from the
quantum-inspired covariance machinery, which remains inert or actively
harmful throughout. A real quantum circuit realization (Qiskit) confirms the
formalism matches its classical simulation exactly and characterizes its
NISQ cost: hundreds of Pauli measurement terms per quantity, cut 4-12x by
standard measurement grouping.

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
classical-limit reduction proof (`tests/test_operator.py`), the
circuit-vs-classical exact-match check (`tests/test_circuit.py`, skipped if
Qiskit is not installed), and smoke tests for all 9 baselines
(`tests/test_baselines.py`).

## Repository structure

```
src/quantum_al/       the installable package: the real, tested formalism
  operator.py            core covariance-aware formalism (Eq. 1-6)
  operator_v2.py         two failed narrow fix attempts (domain grouping, importance weighting)
  operator_v3.py         the residual-coupled fix that reaches parity
  circuit.py             real Qiskit circuit realization + NISQ resource tools
  baselines.py           9 classical active-learning acquisition strategies
  data_utils.py          load real Materials Project data
  fetch_data.py          (re)fetch data from the Materials Project API

benchmarks/            runnable scripts that produced every table/figure
  run_primary_benchmark.py       Tables III/IV: primary comparison + significance
  run_improvement_attempt.py     the two failed narrow fixes
  run_v3_test.py / run_v3_ablation.py / run_v3_all_tasks.py   the residual-coupled fix + its ablation
  run_quantum_circuit_experiment.py   NISQ feasibility characterization
  make_paper_figures.py          regenerates figures/*.pdf from results/*.json

tests/                 pytest suite, run in CI on every push
papers/                 full research manuscripts (IEEE-conference-style and
                        npj-Computational-Materials-style versions) plus
                        superseded prior drafts and reviewer feedback, kept
                        for provenance
figures/                figures embedded in the papers, generated from results/
results/                raw JSON output backing every number in the papers (gitignored; regenerate via benchmarks/)
data/                   real Materials Project data (gitignored; regenerate via src/quantum_al/fetch_data.py)
```

## Regenerating the data and results

```bash
export MP_API_KEY=your_materials_project_api_key   # https://next-gen.materialsproject.org/api
python -m quantum_al.fetch_data
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
