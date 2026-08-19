"""Loads Materials Project JSON files from data/ (see fetch_data.py) into
(X, y, meta) arrays. 21 feature columns: 6 structural/summary descriptors
+ 15 composition-derived stats. crystal_system is metadata only for
regression tasks (it's the target for the classification task).
"""
import json
import os

import numpy as np

# <repo_root>/src/quantum_al/data_utils.py -> up three levels to <repo_root>
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "data")

FEATURE_COLUMNS = [
    "nelements", "density", "volume_per_atom", "nsites", "energy_above_hull",
    "space_group_number",
    "X_mean", "X_std", "X_range",
    "atomic_radius_mean", "atomic_radius_std", "atomic_radius_range",
    "atomic_mass_mean", "atomic_mass_std", "atomic_mass_range",
    "row_mean", "row_std", "row_range",
    "group_mean", "group_std", "group_range",
]

REGRESSION_TASKS = [
    "band_gap", "formation_energy", "bulk_modulus",
    "magnetic_moment", "dielectric_constant",
]
CLASSIFICATION_TASK = "crystal_system"


def load_task(name, drop_na=True):
    """(X, y, meta) for a task JSON in data/. y is float, except str labels
    for crystal_system."""
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, "r") as f:
        rows = json.load(f)

    X_list, y_list, meta = [], [], []
    for row in rows:
        if drop_na:
            if any(row.get(c) is None for c in FEATURE_COLUMNS):
                continue
            if row.get("target") is None:
                continue
        feat = [row.get(c, np.nan) for c in FEATURE_COLUMNS]
        X_list.append(feat)
        y_list.append(row["target"])
        meta.append({
            "material_id": row.get("material_id"),
            "formula_pretty": row.get("formula_pretty"),
            "crystal_system": row.get("crystal_system"),
        })

    X = np.asarray(X_list, dtype=float)
    if name == CLASSIFICATION_TASK:
        y = np.asarray(y_list)
    else:
        y = np.asarray(y_list, dtype=float)
    return X, y, meta


def standardize(X_train, *others):
    """Fit mean/std on X_train, apply to X_train and any other arrays."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    out = [(X_train - mu) / sigma]
    for X in others:
        out.append((X - mu) / sigma)
    return tuple(out) if len(out) > 1 else out[0]


if __name__ == "__main__":
    for t in REGRESSION_TASKS:
        X, y, meta = load_task(t)
        print(t, X.shape, y.shape, "target range:", y.min(), y.max())
    X, y, meta = load_task(CLASSIFICATION_TASK)
    print(CLASSIFICATION_TASK, X.shape, y.shape, "classes:", sorted(set(y.tolist())))
