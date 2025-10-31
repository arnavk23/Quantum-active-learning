# Quantum-Enhanced Active Learning for Materials Discovery

## Overview

This repository contains the complete implementation and research paper for "Quantum-Enhanced Active Learning for Accelerated Materials Discovery: A Novel Framework Combining Quantum Superposition and Multi-Observable Uncertainty Quantification" - a groundbreaking approach that leverages quantum computing principles to revolutionize materials science.

## Key Achievements

### Research Impact

- **35% reduction** in required experiments for materials discovery
- **Statistically significant improvements** (p < 0.01) over 9 state-of-the-art methods
- **First quantum-enhanced active learning framework** specifically designed for materials science
# Quantum-Enhanced Active Learning for Materials Discovery

This repository contains the LaTeX source, code, and supporting materials for the manuscript "Quantum-Enhanced Active Learning for Accelerated Materials Discovery." The project develops a quantum-inspired representation and a multi-observable uncertainty quantification strategy that improves sample efficiency in materials discovery workflows.

Below you'll find a concise, practical README: how to build the paper, run the helper scripts, and reproduce the experiments at a high level.

## Repository contents (short)

- `quantum_paper_fixed.tex` — LaTeX manuscript source.
- `quantum_paper_fixed.pdf` — Compiled PDF (artifact).
- `create_and_compile.sh`, `compile_paper.sh` — helper scripts to build the PDF.
- `scripts/` — Python scripts for preprocessing, training, evaluation, and visualization.
- `models/`, `results/` — (optional) model artifacts and experiment outputs.
- `requirements.txt` — Python dependencies for the scripts.
- `README.md` — this file.

If you need a complete tree view, run `ls -R` from the repository root.

## Quick start — build the paper

These steps assume a Debian/Ubuntu-like environment. If you use macOS or Windows, adapt the package installation accordingly.

1) Install LaTeX (recommended: TeX Live full distribution):

```bash
sudo apt update
sudo apt install -y texlive-full
```

2) Build the PDF using the provided script:

```bash
chmod +x create_and_compile.sh
./create_and_compile.sh
```

3) The script produces `quantum_paper_fixed.pdf` in the repository root. If you prefer manual steps:

```bash
pdflatex quantum_paper_fixed.tex
pdflatex quantum_paper_fixed.tex  # run a second pass to resolve references
```

Notes
- The paper compiles with TeX Live; some systems may warn about pgfplots or font messages — these are typically non-fatal.
- If you encounter a "Missing $ inserted" or grouping error, inspect the `.log` file; I have fixed known issues in the current source.

## Python environment and running scripts

Several scripts under `scripts/` perform preprocessing, training, benchmarking, and plotting. To run them reproducibly:

1) Create and activate a Python venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Example: run a preprocessing or training script

```bash
python scripts/train_surrogate.py --help
python scripts/download_preprocess_mp.py  # if you need to download dataset assets
```

Each script includes a docstring or `--help` output describing inputs and outputs. Many scripts support a small / sample mode so you can run a quick smoke test without large datasets.

If you want, I can create a `Dockerfile` or `devcontainer.json` to pin the environment and make runs fully reproducible.

## Generate the graphical abstract

There is a helper script that produces a graphical abstract PNG (and attempts an SVG):

```bash
python scripts/generate_graphical_abstract.py
```

The script writes `graphical_abstract.png` to the repository root. If you prefer a true vector SVG I can convert or re-create the figure using an SVG library.

## Reproducibility and experiments

- The experiments in the paper were run with fixed seeds and the evaluation protocol described in Section `sec:exp` of the manuscript. When available, model checkpoints, logs, and seeds are stored under `results/`.
- `requirements.txt` lists the Python packages required by the scripts. Use a virtual environment or container to isolate runs.
- For large datasets, the preprocessing scripts include deterministic imputation and standardization routines so reported results are reproducible.

If you need me to (a) run the experiments end to end, (b) prepare a minimal reproducible example, or (c) produce a Docker image, say which option and I'll proceed.

## Structure and important scripts

- `create_and_compile.sh` — wrapper for LaTeX compilation (multi-pass, cleans aux files).
- `scripts/download_preprocess_mp.py` — dataset download and preprocessing helper.
- `scripts/train_surrogate.py` — trains surrogate models used in the active learning loop.
- `scripts/evaluate_benchmark.py` — runs benchmark comparisons against baselines.
- `scripts/generate_graphical_abstract.py` — creates the graphical abstract PNG/SVG.

Open any script to see usage examples and flags.

## How to cite / contact

If you use the methods or code here, please cite the manuscript once it is available. For questions, reproducibility requests, or collaboration, contact:

Arnav Kapoor — arnavkapoor23@iiserb.ac.in

## License

This repository is provided for academic and research use. See `LICENSE` for details.