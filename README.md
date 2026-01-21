# Quantum-Enhanced Active Learning for Materials Discovery

This repository contains the complete implementation and research paper for "Quantum-Enhanced Active Learning for Accelerated Materials Discovery: A Novel Framework Combining Quantum Superposition and Multi-Observable Uncertainty Quantification".

### Key Achievements

- **35% reduction** in required experiments for materials discovery
- **Statistically significant improvements** (p < 0.01) over 9 state-of-the-art methods
- **First quantum-enhanced active learning framework** specifically designed for materials science

### Python environment and running scripts

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

### Contact

If you use the methods or code here, please cite the manuscript once it is available. For questions, reproducibility requests, or collaboration, contact: Arnav Kapoor — arnavkapoor23@iiserb.ac.in
