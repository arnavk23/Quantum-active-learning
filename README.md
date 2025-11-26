# Quantum-Enhanced Active Learning for Materials Discovery

This repository contains the complete implementation and research paper for "Quantum-Enhanced Active Learning for Accelerated Materials Discovery: A Novel Framework Combining Quantum Superposition and Multi-Observable Uncertainty Quantification" - a groundbreaking approach that leverages quantum computing principles to revolutionize materials science.

### Key Achievements

- **35% reduction** in required experiments for materials discovery
- **Statistically significant improvements** (p < 0.01) over 9 state-of-the-art methods
- **First quantum-enhanced active learning framework** specifically designed for materials science

### Repository contents (short)

- `quantum_paper_fixed.tex` — LaTeX manuscript source.
- `scripts/` — Python scripts for preprocessing, training, evaluation, and visualization.
- `models/`, `results/` — model artifacts and experiment outputs.
- `requirements.txt` — Python dependencies for the scripts.

### Python environment and running scripts

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

### Reproducibility and experiments

- The experiments in the paper were run with fixed seeds and the evaluation protocol described in Section `sec:exp` of the manuscript. When available, model checkpoints, logs, and seeds are stored under `results/`.
- `requirements.txt` lists the Python packages required by the scripts. Use a virtual environment or container to isolate runs.
- For large datasets, the preprocessing scripts include deterministic imputation and standardization routines so reported results are reproducible.

### Contact

If you use the methods or code here, please cite the manuscript once it is available. For questions, reproducibility requests, or collaboration, contact: Arnav Kapoor — arnavkapoor23@iiserb.ac.in

### License

This repository is provided for academic and research use. See `LICENSE` for details.
