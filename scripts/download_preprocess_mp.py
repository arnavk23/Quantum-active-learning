
from mp_api.client import MPRester
import pandas as pd
import os

API_KEY = "VYHVfA6CFvAep4e8ddobao7RAijtG4Tp"
DATA_DIR = "./data"
OUTPUT_FILE = os.path.join(DATA_DIR, "mp_dft_data.csv")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    with MPRester(API_KEY) as mpr:
        results = mpr.materials.search(
            elements=["Li", "Fe", "O"],
            fields=[
                "material_id", "composition", "structure", "nsites",
                "volume", "density", "density_atomic"
            ]
        )
    df = pd.DataFrame(results)
    print("Columns in DataFrame:", df.columns)
    # Print first row to inspect available keys
    if not df.empty:
        print("First row:", df.iloc[0])
    # Save raw data for inspection
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()