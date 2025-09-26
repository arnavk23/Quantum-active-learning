"""
Extracts nsites, volume, density, density_atomic from the raw mp_dft_data.csv and saves a clean CSV for ML.
"""
import pandas as pd
import re
import os

RAW_PATH = "./data/mp_dft_data.csv"
CLEAN_PATH = "./data/mp_dft_data_clean.csv"

# Helper to extract value from tuple string
pattern = re.compile(r"\('(.*?)', (.*?)\)")

def extract_value(row, key):
    for item in row:
        match = pattern.match(item)
        if match and match.group(1) == key:
            val = match.group(2)
            try:
                return float(val)
            except:
                return None
    return None

def main():
    df_raw = pd.read_csv(RAW_PATH, header=None)
    # Each row is a list of tuple strings
    clean_data = {
        "nsites": [],
        "volume": [],
        "density": [],
        "density_atomic": []
    }
    for _, row in df_raw.iterrows():
        row = row.values
        clean_data["nsites"].append(extract_value(row, "nsites"))
        clean_data["volume"].append(extract_value(row, "volume"))
        clean_data["density"].append(extract_value(row, "density"))
        clean_data["density_atomic"].append(extract_value(row, "density_atomic"))
    df_clean = pd.DataFrame(clean_data)
    df_clean = df_clean.dropna().drop_duplicates()
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH} with {len(df_clean)} entries.")

if __name__ == "__main__":
    main()
