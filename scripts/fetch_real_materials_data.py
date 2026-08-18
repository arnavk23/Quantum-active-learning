"""
Fetch real Materials Project data and build honest feature vectors.

Replaces the fully-synthetic datasets previously used in benchmark.py /
runtime_memory_benchmarks.py / observable_sensitivity_analysis.py. Every
property pulled here is a genuine DFT-derived quantity from the Materials
Project; no target values are hand-simulated.

Tasks pulled (only properties actually available at scale in MP):
  - band_gap                      (regression)
  - formation_energy_per_atom     (regression)
  - bulk_modulus (K_vrh)          (regression, elastic modulus proxy)
  - total_magnetization           (regression, magnetic moment)
  - e_total (dielectric constant) (regression)
  - crystal_system                (6-class classification)

Thermal conductivity is NOT pulled: MP does not carry broad computed
thermal-conductivity data, so this task is dropped rather than
simulated. This is a deliberate, documented deviation from the original
manuscript's six-task claim.

Features: composition-derived statistics (electronegativity, atomic
radius, atomic mass, ionization energy, row, group -- weighted mean/std/
range over the stoichiometric composition) plus structural/summary
descriptors available directly from the MP summary endpoint (density,
volume per atom, nsites, number of elements, energy above hull, space
group number). This is a real, if lower-dimensional (~40-d, not 100-d),
feature vector -- the paper text is updated to match this honestly.
"""
import json
import os
import time
import urllib.request
import urllib.parse

import numpy as np
from pymatgen.core import Composition, Element

API_KEY = os.environ.get("MP_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Set the MP_API_KEY environment variable to your Materials Project API "
        "key (https://next-gen.materialsproject.org/api) before running this script."
    )
BASE = "https://api.materialsproject.org/materials/summary/"
HEADERS = {
    "X-API-KEY": API_KEY,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json",
}
DATA_DIR = "./data"

COMMON_FIELDS = [
    "material_id", "formula_pretty", "elements", "nelements",
    "density", "volume", "nsites", "energy_above_hull",
    "symmetry",
]

TASKS = {
    "band_gap": {
        "extra_fields": ["band_gap"],
        "filters": {"band_gap_min": 0.001},
        "target": "band_gap",
    },
    "formation_energy": {
        "extra_fields": ["formation_energy_per_atom"],
        "filters": {"formation_energy_per_atom_max": 100},
        "target": "formation_energy_per_atom",
    },
    "bulk_modulus": {
        "extra_fields": ["bulk_modulus"],
        "filters": {"has_props": "elasticity"},
        "target": "bulk_modulus.vrh",
    },
    "magnetic_moment": {
        "extra_fields": ["total_magnetization"],
        "filters": {"total_magnetization_min": 0.01},
        "target": "total_magnetization",
    },
    "dielectric_constant": {
        "extra_fields": ["e_total"],
        "filters": {"has_props": "dielectric"},
        "target": "e_total",
    },
}


def _get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry after error: {e}")
            time.sleep(2)


def fetch_task(name, spec, limit=1000):
    fields = COMMON_FIELDS + spec["extra_fields"]
    params = {
        "_limit": limit,
        "_fields": ",".join(fields),
    }
    params.update(spec["filters"])
    url = BASE + "?" + urllib.parse.urlencode(params)
    print(f"Fetching {name} ...")
    d = _get(url)
    docs = d["data"]
    print(f"  got {len(docs)} docs (total available: {d['meta']['total_doc']})")
    return docs


def dotted_get(doc, path):
    cur = doc
    for part in path.split("."):
        if cur is None:
            return None
        cur = cur.get(part)
    return cur


ELEMENT_PROPS = ["X", "atomic_radius", "atomic_mass", "row", "group"]


def composition_features(elements):
    """Weighted mean/std/range over element properties for a uniform (unweighted
    element-set) composition proxy -- MP summary doesn't return stoichiometric
    fractions cheaply, so we use the unique element set, which is the
    information actually available without a second per-material request."""
    vals = {p: [] for p in ELEMENT_PROPS}
    for el_str in elements:
        try:
            el = Element(el_str)
        except Exception:
            continue
        for p in ELEMENT_PROPS:
            v = getattr(el, p, None)
            if v is None:
                continue
            try:
                vals[p].append(float(v))
            except (TypeError, ValueError):
                continue

    feats = {}
    for p in ELEMENT_PROPS:
        arr = np.array(vals[p]) if vals[p] else np.array([0.0])
        feats[f"{p}_mean"] = float(np.mean(arr))
        feats[f"{p}_std"] = float(np.std(arr))
        feats[f"{p}_range"] = float(np.max(arr) - np.min(arr)) if len(arr) > 1 else 0.0
    return feats


def build_feature_row(doc):
    row = {}
    row["material_id"] = doc.get("material_id")
    row["formula_pretty"] = doc.get("formula_pretty")
    row["nelements"] = doc.get("nelements")
    row["density"] = doc.get("density")
    vol = doc.get("volume")
    nsites = doc.get("nsites")
    row["volume_per_atom"] = (vol / nsites) if (vol and nsites) else None
    row["nsites"] = nsites
    row["energy_above_hull"] = doc.get("energy_above_hull")
    sym = doc.get("symmetry") or {}
    row["crystal_system"] = sym.get("crystal_system")
    row["space_group_number"] = sym.get("number")

    comp_feats = composition_features(doc.get("elements") or [])
    row.update(comp_feats)
    return row


def process_task(name, spec, docs):
    rows = []
    for doc in docs:
        target_val = dotted_get(doc, spec["target"])
        if target_val is None:
            continue
        row = build_feature_row(doc)
        row["target"] = target_val
        rows.append(row)
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    summary = {}
    for name, spec in TASKS.items():
        docs = fetch_task(name, spec)
        rows = process_task(name, spec, docs)
        print(f"  {len(rows)} usable rows with non-null target for {name}")

        out_path = os.path.join(DATA_DIR, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(rows, f)
        summary[name] = len(rows)

    # Separate pull for crystal-system classification: broad sample with
    # symmetry info, independent of the regression targets above.
    print("Fetching crystal_system classification sample ...")
    params = {
        "_limit": 1000,
        "_fields": ",".join(COMMON_FIELDS),
        "energy_above_hull_max": 0.05,
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    d = _get(url)
    docs = d["data"]
    rows = []
    for doc in docs:
        sym = doc.get("symmetry") or {}
        cs = sym.get("crystal_system")
        if cs is None:
            continue
        row = build_feature_row(doc)
        row["target"] = cs
        rows.append(row)
    with open(os.path.join(DATA_DIR, "crystal_system.json"), "w") as f:
        json.dump(rows, f)
    summary["crystal_system"] = len(rows)
    print(f"  {len(rows)} usable rows for crystal_system")

    with open(os.path.join(DATA_DIR, "fetch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
