# 0. Import
import os
import sys
import time
import platform
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import cleaning

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# 1. Config
SEED = 42
PIPELINE = "manual"
RESULTS_FILE = "results/manual_results.csv"

DATASETS = {
    "bank_marketing": {"target": "y"},
    "online_shoppers": {"target": "Revenue"},
    "credit_card": {"target": "default.payment.next.month"}
}

CONDITIONS = [
    "clean", "missing_mild", "missing_severe", "outlier_mild", "outlier_severe", 
    "noise_mild" , "noise_severe"
]

# 1b. Loading function
def load_and_clean(ds_name, cond, target):
    train = pd.read_csv(f"data/messy/{ds_name}/{cond}/train.csv")
    test = pd.read_csv(f"data/processed/{ds_name}/test.csv")

    rules = cleaning.VALIDITY_RULES[ds_name]
    t0 = time.perf_counter()
    train, report = cleaning.clean_all(train, target, rules)
    cleaning_time = time.perf_counter() - t0

    return train, test, report, cleaning_time

# 2. Feature Engineering
def bank_features(df):
    df = df.copy()
    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    return df

def shoppers_features(df):
    df = df.copy()
    df["has_page_values"] = (df["PageValues"] > 0).astype(int)
    for col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[col] = df[col].round().astype(int).astype(str)
    return df


def credit_features(df):
    df = df.copy()
    df["utilization"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["utilization"] = df["utilization"].fillna(0)
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        df[col] = df[col].round().astype(int).astype(str)
    return df


# 3. Modeling
def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

def get_models():
    return {
        "logistic_regression": LogisticRegression(
            class_weight = "balanced", max_iter=1000, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED
        )
    }

def best_threshold(y_true, proba):
    best_t = 0.5
    best_f1 = 0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def select_and_fit(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_name = None
    best_cv_f1 = -1
    best_t = 0.5

    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        t, f1 = best_threshold(y, proba)
        print(f"{name:<24} CV F1={f1:.4f} threshold={t:.2f}")
        if f1 > best_cv_f1:
            best_name = name
            best_cv_f1 = f1
            best_t = t

    final = Pipeline([("prep", build_preprocessor(X)), ("model", get_models()[best_name])])
    final.fit(X, y)
    return final, best_name, best_cv_f1, best_t


# 4. Run one dataset / condition
FEATURE_FUNCS = {
    "bank_marketing":  bank_features,
    "online_shoppers": shoppers_features,
    "credit_card":     credit_features,
}


def run_one(ds_name, cond, target):
    # load + clean (same cleaning as Hybrid)
    train, test, report, cleaning_time = load_and_clean(ds_name, cond, target)

    # feature engineering on BOTH train and test
    add_features = FEATURE_FUNCS[ds_name]
    train = add_features(train)
    test = add_features(test)

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    # model selection + fit
    t0 = time.perf_counter()
    model, model_name, cv_f1, threshold = select_and_fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # predict on clean test with the chosen threshold
    t0 = time.perf_counter()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    predict_time = time.perf_counter() - t0

    return {
        "pipeline": PIPELINE,
        "dataset": ds_name,
        "condition": cond,
        "run": 1,
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "cleaning_time_s": round(cleaning_time, 1),
        "fit_time_s": round(fit_time, 1),
        "predict_time_s": round(predict_time, 2),
        "best_model": model_name,
        "cv_f1": round(cv_f1, 4),
        "decision_threshold": round(threshold, 2),
        "values_invalid": report["values_invalid"],
        "values_imputed": report["values_imputed"],
        "rows_removed_label_noise": report["rows_removed_label_noise"],
        "n_train": len(train),
        "n_test": len(test),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# 5. Saving + main
def already_done():
    done = []
    if os.path.exists(RESULTS_FILE):
        results = pd.read_csv(RESULTS_FILE)
        for _, r in results.iterrows():
            done.append((r["dataset"], r["condition"]))
    return done


def save_result(row):
    os.makedirs("results", exist_ok=True)
    first_time = not os.path.exists(RESULTS_FILE)
    pd.DataFrame([row]).to_csv(RESULTS_FILE, mode="a", index=False, header=first_time)


def main():
    datasets = list(DATASETS)
    conditions = CONDITIONS
    if len(sys.argv) >= 2:
        datasets = [sys.argv[1]]
    if len(sys.argv) >= 3:
        conditions = [sys.argv[2]]

    done = already_done()

    for ds_name in datasets:
        target = DATASETS[ds_name]["target"]
        for cond in conditions:
            if (ds_name, cond) in done:
                print(f"{ds_name} / {cond}: already done, skipping")
                continue

            print(f"\n{ds_name} / {cond}")
            try:
                row = run_one(ds_name, cond, target)
                save_result(row)
                print(f"    {row['best_model']}  F1={row['f1']:.4f}  AUC={row['auc']:.4f}")
            except Exception as e:
                print(f"    FAILED: {e}")


if __name__ == "__main__":
    main()