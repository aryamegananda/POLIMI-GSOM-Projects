# 0. Import
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from cleanlab.filter import find_label_issues
import sys
sys.path.append("../src")
from prepare_data import DATASETS


# 1. Functions
# 1a. Outliers Handling


def validity_check(train, rules):
    train = train.copy()
    for col in rules:
        low, high = rules[col]
        if low is not None:
            train.loc[train[col] < low, col] = np.nan
        if high is not None:
            train.loc[train[col] > high, col] = np.nan
    return train
    

# 1b. Missing value
def clean_missing(train, target):
    train = train.copy()

    # numeric columns (except target) and categorical columns
    num_cols = []
    cat_cols = []
    for col in train.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and not pd.api.types.is_bool_dtype(train[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    # nothing missing -> nothing to do
    if train[num_cols].isnull().sum().sum() == 0:
        return train

    # categoricals one-hot encoded so MICE can use them as extra information
    X = pd.get_dummies(train[num_cols + cat_cols], columns=cat_cols, dtype=float)

    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=train.index)

    # only the numeric columns are replaced; categoricals stay as they were
    train[num_cols] = X_imputed[num_cols]
    return train


# 1c. Handle label noise
def handle_label_noise(train, target):
    train = train.copy()

    X = pd.get_dummies(train.drop(columns=[target]), dtype=float)
    y = train[target].values

    # out-of-sample probabilities from a simple, fixed model (5-fold CV)
    model = HistGradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

    # True = label probably wrong
    issues = find_label_issues(labels=y, pred_probs=pred_probs, frac_noise=0.5)

    train = train[~issues].reset_index(drop=True)
    return train

# 1d. Clean
def fix_imputed_values(train, codes, bounds):
    train = train.copy()
    for col in codes:
        train[col] = train[col].round()
    for col in bounds:
        lo, hi = bounds[col]
        train[col] = train[col].clip(lower=lo, upper=hi)
    return train

# 1e. Clean
def clean_all(train, ds_name):
    info = DATASETS[ds_name]
    target = info["target"]
    bounds = info["bounds"]
    codes = info["codes"]

    report = {}
    report["rows_before"] = len(train)
    report["pos_rate_before"] = round(train[target].mean(), 4)
    nan_before = int(train.isnull().sum().sum())

    train = validity_check(train, bounds)
    report["values_invalid"] = int(train.isnull().sum().sum() - nan_before)
    report["values_imputed"] = int(train.isnull().sum().sum())

    train = clean_missing(train, target)
    train = fix_imputed_values(train, codes, bounds)

    rows_before_noise = len(train)
    train = handle_label_noise(train, target)
    report["rows_removed_label_noise"] = rows_before_noise - len(train)
    report["rows_after"] = len(train)
    report["pos_rate_after"] = round(train[target].mean(), 4)

    return train, report


# 2. Run
def main():
    import os
    from prepare_data import CONDITIONS

    report_rows = []
    for ds_name, info in DATASETS.items():
        print(f"\n{'=' * 60}\n{ds_name.upper()}\n{'=' * 60}")

        for cond in CONDITIONS:
            train = pd.read_csv(f"data/messy/{ds_name}/{cond}/train.csv")
            cleaned, report = clean_all(train, ds_name)

            out_dir = f"data/cleaned/{ds_name}/{cond}"
            os.makedirs(out_dir, exist_ok=True)
            cleaned.to_csv(f"{out_dir}/train.csv", index=False)

            # checks
            nan_left = int(cleaned.isnull().sum().sum())
            bound_breaks = 0
            for col in info["bounds"]:
                lo, hi = info["bounds"][col]
                if lo is not None:
                    bound_breaks += int((cleaned[col] < lo).sum())
                if hi is not None:
                    bound_breaks += int((cleaned[col] > hi).sum())
            decimal_codes = 0
            for col in info["codes"]:
                decimal_codes += int((cleaned[col] != cleaned[col].round()).sum())

            print(f"\n  {cond}: {report}")
            print(f"    NaN left: {nan_left} | bound breaks: {bound_breaks} | decimal codes: {decimal_codes}")

            report["dataset"] = ds_name
            report["condition"] = cond
            report_rows.append(report)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(report_rows).to_csv("results/cleaning_report.csv", index=False)
    print("\nSaved results/cleaning_report.csv")


if __name__ == "__main__":
    main()