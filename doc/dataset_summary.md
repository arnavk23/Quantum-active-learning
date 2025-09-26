# DFT and Quantum Materials Datasets for ML Surrogate Modeling

This document summarizes the top open datasets for DFT energies/forces and quantum materials, suitable for machine learning surrogates, uncertainty quantification, and quantum computing research.

---

## 1. Materials Project (MP)
- **Website:** https://materialsproject.org/
- **Content:** DFT-calculated properties (energies, forces, band structures, etc.) for thousands of materials.
- **Access:** Free with registration; data available via web interface and API.
- **Format:** JSON, CSV, and Python tools (pymatgen).
- **Suitability:** Gold standard for ML surrogates, UQ, and quantum materials research.
- **How to Use:**
  - Register for an account to access the API.
  - Use the [pymatgen](https://pymatgen.org/) Python library for programmatic access and data analysis.
  - Download datasets for bulk analysis or ML training.

---

## 2. OQMD (Open Quantum Materials Database)
- **Website:** https://oqmd.org/
- **Content:** DFT thermodynamic and structural properties for 1.3M+ materials.
- **Access:** Free, CC-BY 4.0 license; bulk download and RESTful API (OPTIMADE).
- **Format:** CSV, API, web queries.
- **Suitability:** Large-scale, diverse DFT data for ML, UQ, and benchmarking.
- **How to Use:**
  - Use the [Download page](https://oqmd.org/download/) for bulk data.
  - Query via [API](https://oqmd.org/api) or [OPTIMADE](https://oqmd.org/optimade) for custom searches.
  - Integrate with ML workflows using CSV or API outputs.

---

## 3. NOMAD (Novel Materials Discovery Laboratory)
- **Website:** https://nomad-coe.eu/
- **Content:** Repository for raw and processed DFT data from many codes and projects.
- **Access:** Free, open source; web portal and API.
- **Format:** Various (HDF5, CSV, JSON); supports high-throughput and exascale workflows.
- **Suitability:** Excellent for custom ML datasets, quantum materials, and reproducibility.
- **How to Use:**
  - Search and download datasets via the [NOMAD Archive](https://nomad-lab.eu/prod/v1/gui/search/).
  - Use the [NOMAD AI Toolkit](https://nomad-lab.eu/aitoolkit) for ML and data analysis.
  - Access data programmatically via [API](https://nomad-lab.eu/prod/v1/api.html).

---

## 4. AFLOW
- **Website:** https://aflow.org/
- **Content:** 3.9M+ compounds, 800M+ calculated properties (DFT energies, band structures, etc.).
- **Access:** Free; web search, API, and ML tools.
- **Format:** API, CSV, JSON.
- **Suitability:** Massive database for ML, UQ, and quantum computing applications.
- **How to Use:**
  - Search for materials and properties via the [AFLOW Search](https://aflow.org/search).
  - Use the [AFLOW REST API](https://aflow.org/documentation) for programmatic access.
  - Download data for ML model training and benchmarking.

---

## Recommended Tools
- **pymatgen**: Python library for MP data access and analysis.
- **ASE (Atomic Simulation Environment)**: For working with DFT data and running simulations.
- **NOMAD AI Toolkit**: For ML and data analysis on NOMAD datasets.