# Project Documentation: ML Surrogate Modeling for DFT with UQ, Misspecification Detection, and Quantum Computing

## Overview
This project demonstrates a reproducible pipeline for building machine learning (ML) surrogates for Density Functional Theory (DFT) energies/forces, integrating uncertainty quantification (UQ), misspecification detection, and quantum computing techniques. The workflow includes data acquisition, preprocessing, model training, UQ, OOD detection, benchmarking, and advanced quantum computing demos.

## Folder Structure
- `data/`: Raw and cleaned DFT datasets (e.g., Materials Project)
- `scripts/`: Python scripts for data processing, ML training, UQ, OOD detection, benchmarking, and quantum computing
- `models/`: Saved ML models (Random Forest, ensemble)
- `results/`: Visualizations, metrics, and output files
- `doc/`: Literature review, implementation plan, dataset summary, and documentation

## Pipeline Summary
1. **Data Acquisition & Preprocessing**
   - Download DFT data (e.g., from Materials Project)
   - Clean and extract relevant features
2. **ML Surrogate Training**
   - Train Random Forest and ensemble models for energy/force prediction
   - Save models and evaluation metrics
3. **Uncertainty Quantification (UQ)**
   - Use ensemble variance for UQ
   - Visualize prediction uncertainty
4. **Misspecification Detection (OOD)**
   - Flag high-uncertainty predictions as OOD
   - Save OOD flags and visualize uncertainty
5. **Benchmarking & Evaluation**
   - Compare surrogate predictions to DFT
   - Assess UQ calibration and fallback triggers
6. **Quantum Computing Integration**
   - Advanced script demonstrates quantum circuit basics, algorithms (Grover, VQE, QPE), and hybrid quantum-classical ML
   - Visualizations and error analysis included

## Key Scripts
- `download_preprocess_mp.py`: Download and preprocess DFT data
- `clean_mp_csv.py`: Extract features for ML
- `train_surrogate.py`: Train baseline ML surrogate
- `train_surrogate_with_uq.py`: Train ensemble for UQ
- `misspecification_detection.py`: OOD detection and fallback
- `evaluate_benchmark.py`: Benchmark surrogate and UQ
- `quantum_angle_demo.py`: Quantum computing and hybrid ML demo

## Reproducibility & References
- All scripts are documented and reproducible
- See `doc/literature_review_detailed.md` for academic context
- Dataset details in `doc/dataset_summary.md`
- Implementation plan in `doc/implementation_plan.md`

## How to Run
1. Install dependencies (see README or use virtualenv)
2. Run scripts in order for full pipeline
3. Visualizations and results are saved in `results/`

## Contact & Citation
For academic use, cite the literature in `doc/literature_review_detailed.md` and acknowledge this repository.
