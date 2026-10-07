# 0. Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pucktrick.missing import missing
from pucktrick.outliers import outlier
from pucktrick.labels import labels
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import os
import random

# 1. Config
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# 2. Dataset
DATASETS = {
    "bank_marketing": {
        "file": "data/raw/bank-additional-full.csv",
        "sep": ";",
        "target": "y",
        "drop": ["duration"],
        "bool_to_int": [],
        "continuous": ["age", "campaign", "pdays", "previous", "emp.var.rate",
                       "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
        "codes": [],
    },
    "online_shoppers": {
        "file": "data/raw/online_shoppers_intention.csv",
        "sep": ",",
        "target": "Revenue",
        "drop": [],
        "bool_to_int": ["Weekend"],
        "continuous": ["Administrative", "Administrative_Duration", "Informational",
                       "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                       "BounceRates", "ExitRates", "PageValues", "SpecialDay"],
        "codes": ["OperatingSystems", "Browser", "Region", "TrafficType", "Weekend"],
        "bounds": {
            "Administrative": (0, None), "Administrative_Duration": (0, None),
            "Informational": (0, None), "Informational_Duration": (0, None),
            "ProductRelated": (0, None), "ProductRelated_Duration": (0, None),
            "BounceRates": (0, 1), "ExitRates": (0, 1),
            "PageValues": (0, None), "SpecialDay": (0, 1),
            "OperatingSystems": (1, None), "Browser": (1, None),
            "Region": (1, None), "TrafficType": (1, None), "Weekend": (0, 1),
        },
    },
    "credit_card": {
        "file": "data/raw/UCI_Credit_Card.csv",
        "sep": ",",
        "target": "default.payment.next.month",
        "drop": ["ID"],
        "bool_to_int": [],
        "continuous": ["LIMIT_BAL", "AGE",
                       "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                       "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
        "codes": ["SEX", "EDUCATION", "MARRIAGE",
                  "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    },
}

# 3. Set conditions
CONDITIONS = {
    "clean":          {"type": None,      "pct": 0.0},
    "missing_mild":   {"type": "missing", "pct": 0.10},
    "missing_severe": {"type": "missing", "pct": 0.20},
    "outlier_mild":   {"type": "outlier", "pct": 0.05},
    "outlier_severe":  {"type": "outlier", "pct": 0.15},
    "noise_mild":     {"type": "noise",   "pct": 0.05},
    "noise_severe":   {"type": "noise",   "pct": 0.20},
}

# 4. Functions
def encode_target(df, target):
    s = df[target]

    if is_bool_dtype(s) or is_numeric_dtype(s):
        df[target] = s.astype(int)
    else:
        s = s.astype(str).str.strip().str.lower()
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
        unknown = set(s.unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected target values in '{target}': {unknown}")
        df[target] = s.map(mapping).astype(int)

    assert set(df[target].unique()) <= {0, 1}, f"Target '{target}' is not binary"
    return df

def inject_per_column(train_df, cols, pct, puck_function, name):
    degraded = train_df.copy()
    for j, col in enumerate(cols):
        set_seed(SEED + j)
        strategy = {
            "affected_features": [col],
            "selection_criteria": "all",
            "percentage": pct,
            "mode": "new",
            "perturbate_data": {"sampling": "random"},
        }
        err, degraded = puck_function(degraded, strategy)
        if err != 0:
           raise RuntimeError(f"PuckTrick returned err={err} for {name} in column {col}")
    return degraded         

def apply_degradation(train_df, target, deg_type, pct, continuous, codes):
    if deg_type == "missing":
        return inject_per_column(train_df, continuous + codes, pct, missing, "missing")

    elif deg_type == "outlier":
        return inject_per_column(train_df, continuous, pct, outlier, "outlier")

    elif deg_type == "noise":
        strategy = {
            "affected_features": [target],
            "selection_criteria": "all",
            "percentage": pct,
            "mode": "new",
            "perturbate_data": {"sampling": "random"},
        }
        err, degraded = labels(train_df, strategy)
        if err !=  0:
            raise RuntimeError(f"PuckTrick returned err={err} for noise")
        return degraded

    else:
        return train_df.copy()

def share_rows_with_nan(df, cols):
    count = 0
    for i in range(len(df)):
        row = df[cols].iloc[i]
        if row.isna().any():
            count += 1
    return count / len(df)

def share_nan_per_column(df, cols):
    for col in cols:
        share = df[col].isna().mean()
        print(f"{col}: {share:.2%}")

def main():
    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*60}")

        # Load
        df = pd.read_csv(ds_info["file"], sep=ds_info["sep"])
        target = ds_info["target"]
        df = encode_target(df, target)

        for col in ds_info["drop"]:
            df = df.drop(columns=col)

        for col in ds_info["bool_to_int"]:
            df[col] = df[col].astype(int)

        for col in ds_info["continuous"] + ds_info["codes"]:
            if col not in df.columns:
                raise ValueError(f"{ds_name}: column '{col}' not found")

        print(f"  Shape: {df.shape}")
        print(f"  Target balance: {df[target].value_counts(normalize=True).round(3).to_dict()}")

        # Split
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=SEED, stratify=df[target]
        )
        print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")

        # Save test set
        test_dir = f"data/processed/{ds_name}"
        os.makedirs(test_dir, exist_ok=True)
        test_df.to_csv(f"{test_dir}/test.csv", index=False)

        # Save clean baseline
        clean_dir = f"data/messy/{ds_name}/clean"
        os.makedirs(clean_dir, exist_ok=True)
        train_df.to_csv(f"{clean_dir}/train.csv", index=False)

        # Generate degraded conditions
        for cond_name, cond in CONDITIONS.items():
            if cond["type"] is None:
                continue

            print(f"\n  Condition: {cond_name} ({cond['type']} @ {cond['pct']:.0%})")
            set_seed(SEED)
            degraded = apply_degradation(
                train_df.copy(), target, cond["type"], cond["pct"], ds_info["continuous"], ds_info["codes"]
            )

            out_dir = f"data/messy/{ds_name}/{cond_name}"
            os.makedirs(out_dir, exist_ok=True)
            degraded.to_csv(f"{out_dir}/train.csv", index=False)

            # Quick validation
            if cond["type"] == "missing":
                cols = ds_info["continuous"] + ds_info["codes"]
                pct_rows = share_rows_with_nan(degraded, cols)
                print(f"Rows with missing values: {pct_rows:.2%}")
                share_nan_per_column(degraded, cols)

            elif cond["type"] == "outlier":
                cols = ds_info["continuous"]
                changed = degraded[cols] != train_df[cols]
                print(f"Rows with outliers: {changed.any(axis=1).mean():.2%}")
                for col in cols:
                    print(f"{col}: {changed[col].mean():.2%}")
          
            elif cond["type"] == "noise":
                flipped = (degraded[target] != train_df[target]).mean()
                print(f"Actual label flip rate: {flipped:.2%}")

        print(f"\nDone: {ds_name}")

    print(f"\n{'='*60}")
    print("  ALL DATASETS PREPARED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
