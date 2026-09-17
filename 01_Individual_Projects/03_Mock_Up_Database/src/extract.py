# 0. Import
import os
import pandas as pd
from config import RAW_DATA_DIR

# 1. Read csv
def extract_site(filename):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(filepath)
    print(f"Extracted {filename}: {len(df)} rows, {len(df.columns)} columns")
    return df

def extract_all():
    files = []
    for f in os.listdir(RAW_DATA_DIR):
        if f.endswith(".csv"):
            files.append(f)


    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    raw_data = {}
    for f in files:
        site_name = f.replace("_results.csv", "")
        raw_data[site_name] = extract_site(f)

    print(f"\nExtracted {len(raw_data)} sites, {sum(len(df) for df in raw_data.values())} total rows")
    return raw_data

if __name__ == "__main__":
    data = extract_all()
    for site, df in data.items():
        print(f"\n--- {site} ---")
        print(df.head())
        print(f"Columns: {list(df.columns)}")