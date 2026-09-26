"""
Output structure:
  data/processed/<dataset>/test.csv              (shared clean test set)
  data/messy/<dataset>/clean/train.csv            (clean baseline)
  data/messy/<dataset>/missing_mild/train.csv     (10% missing)
  data/messy/<dataset>/missing_severe/train.csv   (20% missing)
  data/messy/<dataset>/outlier_mild/train.csv     (5% outliers)
  data/messy/<dataset>/outlier_severe/train.csv   (15% outliers)
  data/messy/<dataset>/noise_mild/train.csv       (5% label noise)
  data/messy/<dataset>/noise_severe/train.csv     (20% label noise)
"""

# 0. Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pucktrick.missing import missing
from pucktrick.outliers import outlier
from pucktrick.labels import labels
import os

# 1. Config
SEED = 42
np.random.seed(SEED)

# 2. Dataset
DATASETS = {
    "bank_marketing": {
        "file": "data/raw/bank-additional-full.csv",
        "sep": ";",
        "target": "y",
    },
    "online_shoppers": {
        "file": "data/raw/online_shoppers_intention.csv",
        "sep": ",",
        "target": "Revenue",
    },
    "credit_card": {
        "file": "data/raw/UCI_Credit_Card.csv",
        "sep": ",",
        "target": "default.payment.next.month",
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

    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        df[target] = s.astype(int)
    else:
        # covers object, category, and pandas 3.x "str"/StringDtype
        s = s.astype(str).str.strip().str.lower()
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
        unknown = set(s.unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected target values in '{target}': {unknown}")
        df[target] = s.map(mapping).astype(int)

    assert set(df[target].unique()) <= {0, 1}, f"Target '{target}' is not binary"
    return df


def apply_degradation(train_df, target, deg_type, pct):
    num_cols = train_df.select_dtypes(include="number").columns.tolist()
    if target in num_cols:
        num_cols.remove(target)

    if deg_type == "missing":
        strategy = {
            "affected_features": num_cols,
            "selection_criteria": "all",
            "percentage": pct,
            "mode": "new",
            "perturbate_data": {"sampling": "random"},
        }
        err, degraded = missing(train_df, strategy)

    elif deg_type == "outlier":
        strategy = {
            "affected_features": num_cols,
            "selection_criteria": "all",
            "percentage": pct,
            "mode": "new",
            "perturbate_data": {"sampling": "random"},
        }
        err, degraded = outlier(train_df, strategy)

    elif deg_type == "noise":
        strategy = {
            "affected_features": [target],
            "selection_criteria": "all",
            "percentage": pct,
            "mode": "new",
            "perturbate_data": {"sampling": "random"},
        }
        err, degraded = labels(train_df, strategy)

    else:
        degraded = train_df.copy()

    return degraded


def main():
    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*60}")

        # Load
        df = pd.read_csv(ds_info["file"], sep=ds_info["sep"])
        target = ds_info["target"]
        df = encode_target(df, target)

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
                continue  # already saved clean

            print(f"\n  Condition: {cond_name} ({cond['type']} @ {cond['pct']:.0%})")
            degraded = apply_degradation(
                train_df.copy(), target, cond["type"], cond["pct"]
            )

            out_dir = f"data/messy/{ds_name}/{cond_name}"
            os.makedirs(out_dir, exist_ok=True)
            degraded.to_csv(f"{out_dir}/train.csv", index=False)

            # Quick validation
            if cond["type"] == "missing":
                pct_actual = degraded[degraded.columns.difference([target])].isnull().mean().mean()
                print(f"    Actual missing rate: {pct_actual:.2%}")
            elif cond["type"] == "noise":
                flipped = (degraded[target] != train_df[target]).mean()
                print(f"    Actual label flip rate: {flipped:.2%}")

        print(f"\n  ✓ Done: {ds_name}")

    print(f"\n{'='*60}")
    print("  ALL DATASETS PREPARED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
