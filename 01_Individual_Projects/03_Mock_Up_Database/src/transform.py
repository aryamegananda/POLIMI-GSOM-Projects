# 0. Import
import pandas as pd
import numpy as np
from config import SCHEMA, VALID_SITES, VALID_STATUSES, VALID_TEST_TYPES, VALID_UNITS


# 1. Column mappings
COLUMN_MAP = {
    "vimodrone": {
        "SampleID": "sample_id",
        "TestType": "test_type",
        "Result": "result_value",
        "Unit": "unit",
        "test_date":"run_date",
        "Operator":"operator_id",
        "Status": "status"
    },
    "poggibonsi": {
        "cod_campione":"sample_id",
        "tipo_test": "test_type",
        "valore": "result_value",
        "unita": "unit",
        "data_analisi":"run_date",
        "operatore":"operator_id",
        "esito": "status"
    },
    "milan_hq": {
        "Sample_Code": "sample_id",
        "Test": "test_type",
        "Value": "result_value",
        "UNIT": "unit",
        "Date": "run_date",
        "Tech": "operator_id",
        "Result_Status": "status",
    }
}


# 2. Status mapping
STATUS_MAP = {
    "passato": "passed",
    "fallito": "failed",
    "in attesa": "pending",
    "pass": "passed",
    "fail": "failed",
    "failed": "failed",
    "pending": "pending",
    "passed": "passed"
}


# 3. Test types mapping
TEST_TYPE_MAP = {
    "ph": "pH",
    "endotoxin": "endotoxin",
    "sterility": "sterility",
    "bioburden": "bioburden",
    "conductivity": "conductivity",
    "toc": "TOC",
    "microbial limit": "microbial_limit",
    "potency": "potency"
}


# 4. Units mapping
UNIT_MAP = {
    "ph": "pH",
    "eu/ml": "EU/mL",
    "pass/fail": "pass/fail",
    "cfu/ml": "CFU/mL",
    "cfu/g": "CFU/g",
    "us/cm": "uS/cm",
    "ppb": "ppb",
    "mg/ml": "mg/mL",
}


# 5. Rename columns
def rename_columns(df, site_name):
    column_map = COLUMN_MAP[site_name]
    df = df.rename(columns=column_map)
    return df


# 6. Add site codes
def add_site_code(df, site_name):
    site_codes = {
        "vimodrone": "VIM",
        "poggibonsi": "POG",
        "milan_hq": "MIL"
    }
    df["site_code"] = site_codes[site_name]
    return df


# 7. Clean strings
def clean_strings(df):
    for col in ["test_type", "unit", "status", "operator_id"]:
        df[col] = df[col].astype(str).str.strip()
    return df


# 8. Standarize test types
def standardize_test_types(df):
    df["test_type"] = df["test_type"].str.lower().map(TEST_TYPE_MAP)
    return df


# 9. Standarize status
def standardize_statuses(df):
    df["status"] = df["status"].str.lower().map(STATUS_MAP)
    return df


# 10. Standarize unit
def standardize_units(df):
    df["unit"] = df["unit"].str.lower().map(UNIT_MAP)
    return df


# 11. Parse dates
def parse_dates(df):
    df["run_date"] = pd.to_datetime(df["run_date"], format="mixed", dayfirst=True)
    return df


# 12. Clean result values
def clean_result_values(df):
    df.loc[df["result_value"] < 0, "result_value"] = np.nan
    df.loc[df["result_value"] > 10000, "result_value"] = np.nan
    return df


# 13. Remove duplicates
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate rows")
    return df


# 14. Handle missing
def handle_missing(df):
    df = df.dropna(subset=["sample_id"])
    df["operator_id"] = df["operator_id"].replace("nan", "UNKNOWN")
    df["operator_id"] = df["operator_id"].fillna("UNKNOWN")
    return df


# 15. Validate
def validate(df):
    invalid_status = ~df["status"].isin(VALID_STATUSES)
    invalid_test = ~df["test_type"].isin(VALID_TEST_TYPES)

    if invalid_status.any():
        print(f"WARNING: {invalid_status.sum()} rows with invalid status -> set to NaN")
        df.loc[invalid_status, "status"] = np.nan

    if invalid_test.any():
        print(f"WARNING: {invalid_test.sum()} rows with invalid test types -> set to NaN")
        df.loc[invalid_test, "test_type"] = np.nan

    return df


# 16. Apply all transformation steps to 1 site's data
def transform_site(df, site_name):
    print(f"Transforming {site_name}...")
    df = rename_columns(df, site_name)
    df = add_site_code(df,site_name)
    df = clean_strings(df)
    df = standardize_test_types(df)
    df = standardize_statuses(df)
    df = standardize_units(df)
    df = parse_dates(df)
    df = clean_result_values(df)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = validate(df)
    print(f"Done. Finished cleaning {len(df)} rows.")
    return df


# 17. Apply transformation to all
def transform_all(raw_data):
    clean_frames = []
    for site_name, df in raw_data.items():
        clean_df = transform_site(df, site_name)
        clean_frames.append(clean_df)

    combined = pd.concat(clean_frames, ignore_index=True)

    column_order = list(SCHEMA.keys())
    combined = combined[column_order]

    print(f"\nTransform completed! {len(combined)} total clean rows")
    return combined


if __name__ == "__main__":
    from extract import extract_all

    raw_data = extract_all()
    clean_data = transform_all(raw_data)

    print(f"\n--- Clean data sample: ---")
    print(clean_data.head(10))
    print(f"\nNull counts: \n{clean_data.isnull().sum()}")

    # Save
    clean_data.to_csv("data/clean/cleaned_results.csv", index=False)
    print(f"Saved to data/clean/cleaned_results.csv")