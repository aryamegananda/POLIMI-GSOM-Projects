"""
This file is AI generated.
Generate 3 messy CSV files simulating lab exports from different Eurofins sites.
Each site has different column names, date formats, and data quality issues.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# --- Site 1: Vimodrone (relatively clean, but different column names) ---
n_vim = 500
vim_data = {
    "SampleID": [f"VIM-2026-{str(i).zfill(5)}" for i in range(1, n_vim + 1)],
    "TestType": np.random.choice(["pH", "endotoxin", "sterility", "bioburden"], n_vim),
    "Result": np.round(np.random.uniform(0.5, 14.0, n_vim), 3),
    "Unit": np.random.choice(["pH", "EU/mL", "pass/fail", "CFU/mL"], n_vim),
    "test_date": pd.date_range("2025-01-01", periods=n_vim, freq="8h").strftime("%Y-%m-%d"),
    "Operator": np.random.choice(["OP001", "OP002", "OP003", "OP004", "OP005"], n_vim),
    "Status": np.random.choice(["passed", "failed", "pending"], n_vim, p=[0.7, 0.15, 0.15]),
}
vim_df = pd.DataFrame(vim_data)
# Inject issues: scattered missing values
missing_idx = np.random.choice(n_vim, 25, replace=False)
vim_df.loc[missing_idx[:15], "Result"] = np.nan
vim_df.loc[missing_idx[15:], "Operator"] = np.nan

# --- Site 2: Poggibonsi (messy — Italian column names, date format, casing issues) ---
n_pog = 750
pog_data = {
    "cod_campione": [f"POG-2026-{str(i).zfill(5)}" for i in range(1, n_pog + 1)],
    "tipo_test": np.random.choice(["pH", "endotoxin", "sterility", "conductivity", "TOC"], n_pog),
    "valore": np.round(np.random.uniform(0.1, 500.0, n_pog), 2),
    "unita": np.random.choice(["pH", "EU/mL", "pass/fail", "uS/cm", "ppb"], n_pog),
    "data_analisi": pd.date_range("2025-01-01", periods=n_pog, freq="6h").strftime("%d/%m/%Y"),
    "operatore": np.random.choice(["BIANCHI_M", "ROSSI_L", "VERDI_A", "NERI_F", "COLOMBO_S"], n_pog),
    "esito": np.random.choice(["PASSATO", "FALLITO", "IN ATTESA"], n_pog, p=[0.65, 0.2, 0.15]),
}
pog_df = pd.DataFrame(pog_data)
# Inject issues: negative values, whitespace, wrong casing, duplicates
neg_idx = np.random.choice(n_pog, 15, replace=False)
pog_df.loc[neg_idx, "valore"] = pog_df.loc[neg_idx, "valore"] * -1
ws_idx = np.random.choice(n_pog, 20, replace=False)
pog_df.loc[ws_idx, "tipo_test"] = pog_df.loc[ws_idx, "tipo_test"].apply(lambda x: f"  {x}  ")
case_idx = np.random.choice(n_pog, 15, replace=False)
pog_df.loc[case_idx, "unita"] = pog_df.loc[case_idx, "unita"].str.lower()
# Add 30 duplicate rows
dup_idx = np.random.choice(n_pog, 30, replace=False)
pog_df = pd.concat([pog_df, pog_df.iloc[dup_idx]], ignore_index=True)

# --- Site 3: Milan HQ (worst — mixed everything) ---
n_mln = 600
mln_data = {
    "Sample_Code": [f"MLN-2026-{str(i).zfill(5)}" for i in range(1, n_mln + 1)],
    "Test": np.random.choice(["ph", "Endotoxin", "STERILITY", "Microbial Limit", "potency"], n_mln),
    "Value": np.round(np.random.uniform(0.01, 1000.0, n_mln), 4),
    "UNIT": np.random.choice(["pH", "EU/mL", "pass/fail", "CFU/g", "mg/mL"], n_mln),
    "Date": pd.date_range("2025-01-01", periods=n_mln, freq="10h"),
    "Tech": np.random.choice(["T.Smith", "J.Wong", "A.Kumar", "R.Tanaka", None], n_mln),
    "Result_Status": np.random.choice(["Pass", "Fail", "PENDING", "passed", "Failed"], n_mln),
}
mln_df = pd.DataFrame(mln_data)
mln_df["Date"] = mln_df["Date"].astype(str)
# Inject issues: mixed date formats
mixed_dates = np.random.choice(n_mln, 40, replace=False)
for i in mixed_dates[:20]:
    d = pd.Timestamp(mln_df.loc[i, "Date"])
    mln_df.loc[i, "Date"] = d.strftime("%B %d, %Y")  # "March 15, 2026"
for i in mixed_dates[20:]:
    d = pd.Timestamp(mln_df.loc[i, "Date"])
    mln_df.loc[i, "Date"] = d.strftime("%d-%m-%Y")    # "15-03-2026"
# Outliers
outlier_idx = np.random.choice(n_mln, 10, replace=False)
mln_df.loc[outlier_idx, "Value"] = np.random.choice([99999.0, -999.0, 88888.0], 10)
# Invalid statuses
invalid_idx = np.random.choice(n_mln, 8, replace=False)
mln_df.loc[invalid_idx, "Result_Status"] = np.random.choice(["APPROVED", "REVIEW", "N/A"], 8)
# Missing values
null_idx = np.random.choice(n_mln, 30, replace=False)
mln_df.loc[null_idx[:20], "Value"] = np.nan
mln_df.loc[null_idx[20:], "UNIT"] = np.nan

# Save
os.makedirs("data/raw", exist_ok=True)
vim_df.to_csv("data/raw/vimodrone_results.csv", index=False)
pog_df.to_csv("data/raw/poggibonsi_results.csv", index=False)
mln_df.to_csv("data/raw/milan_hq_results.csv", index=False)

print("Generated 3 CSV files in data/raw/")
print(f"  vimodrone:   {len(vim_df)} rows")
print(f"  poggibonsi:  {len(pog_df)} rows (includes duplicates)")
print(f"  milan_hq:    {len(mln_df)} rows")