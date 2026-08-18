# Data

This directory holds real Materials Project data pulled by
`scripts/fetch_real_materials_data.py`. It is gitignored (except this file)
because the raw JSON is regenerable and not small.

To regenerate:

```bash
export MP_API_KEY=your_materials_project_api_key   # https://next-gen.materialsproject.org/api
python scripts/fetch_real_materials_data.py
```

This produces `band_gap.json`, `formation_energy.json`, `bulk_modulus.json`,
`magnetic_moment.json`, `dielectric_constant.json`, and `crystal_system.json`,
each ~1000 real materials with the 21-dimensional feature set described in
`scripts/data_utils.py` and used throughout `results/`.
