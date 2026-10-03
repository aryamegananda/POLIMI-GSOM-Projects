# 0. Import
import os
import sys
import time
import shutil
import platform
from datetime import datetime

import pandas as pd
from importlib.metadata import version as pkg_version
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import cleaning

# 1. Config
PIPELINE = "hybrid"
RESULTS_FILE = "results/hybrid_results.csv"
MODELS_DIR = "models/hybrid"
DELETE_MODELS_AFTER_RUN = True

AG_FIT_ARGS = {
    "presets": "best_quality",
    "time_limit": 300,
    "dynamic_stacking": False,
}
AG_EVAL_METRIC = "f1"

# 2. Datasets and conditions
DATASETS = {
    "bank_marketing":  {"target": "y"},
    "online_shoppers": {"target": "Revenue"},
    "credit_card":     {"target": "default.payment.next.month"},
}

CONDITIONS = [
    "clean",
    "missing_mild", "missing_severe",
    "outlier_mild", "outlier_severe",
    "noise_mild", "noise_severe",
]


# 3. Functions
def already_done():
    if not os.path.exists(RESULTS_FILE):
        return set()
    done = pd.read_csv(RESULTS_FILE)
    return set(zip(done["dataset"], done["condition"]))


def append_result(row):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        RESULTS_FILE, mode="a", index=False, header=not os.path.exists(RESULTS_FILE)
    )


def run_one(ds_name, cond, target):
    train = pd.read_csv(f"data/messy/{ds_name}/{cond}/train.csv")
    test = pd.read_csv(f"data/processed/{ds_name}/test.csv")

    # Clean (the only difference from run_automl.py)
    rules = cleaning.VALIDITY_RULES[ds_name]
    t0 = time.perf_counter()
    train_clean, report = cleaning.clean_all(train, target, rules)
    cleaning_time = time.perf_counter() - t0

    model_path = f"{MODELS_DIR}/{ds_name}/{cond}"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # Fit
    t0 = time.perf_counter()
    predictor = TabularPredictor(
        label=target, eval_metric=AG_EVAL_METRIC, path=model_path, verbosity=1
    ).fit(train_clean, **AG_FIT_ARGS)
    fit_time = time.perf_counter() - t0

    # Predict on clean test (test is never cleaned)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    t0 = time.perf_counter()
    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)[predictor.positive_class]
    predict_time = time.perf_counter() - t0

    best_model = predictor.leaderboard(silent=True).iloc[0]["model"]

    row = {
        "pipeline": PIPELINE,
        "dataset": ds_name,
        "condition": cond,
        "run": 1,
        "f1": f1_score(y_test, y_pred, pos_label=predictor.positive_class),
        "auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, pos_label=predictor.positive_class),
        "recall": recall_score(y_test, y_pred, pos_label=predictor.positive_class),
        "cleaning_time_s": round(cleaning_time, 1),
        "fit_time_s": round(fit_time, 1),
        "predict_time_s": round(predict_time, 2),
        "best_model": best_model,
        "positive_class": predictor.positive_class,
        "decision_threshold": getattr(predictor, "decision_threshold", None),
        "values_invalid": report["values_invalid"],
        "values_imputed": report["values_imputed"],
        "rows_removed_label_noise": report["rows_removed_label_noise"],
        "n_train": report["rows_after"],
        "n_test": len(test),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if DELETE_MODELS_AFTER_RUN:
        shutil.rmtree(model_path, ignore_errors=True)

    return row


def main():
    # arguments can be dataset names and/or condition names, e.g.
    # python src/run_hybrid.py bank_marketing
    # python src/run_hybrid.py bank_marketing noise_severe
    args = sys.argv[1:]
    selected_ds = [a for a in args if a in DATASETS] or list(DATASETS)
    selected_cond = [a for a in args if a in CONDITIONS] or CONDITIONS
    unknown = [a for a in args if a not in DATASETS and a not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown argument(s): {unknown}")

    print(f"AutoGluon {pkg_version('autogluon.tabular')} | cleanlab {pkg_version('cleanlab')} | "
          f"Python {platform.python_version()} | {platform.system()} {platform.release()} | "
          f"{platform.processor()}")

    done = already_done()
    total = len(selected_ds) * len(selected_cond)
    i = 0

    for ds_name in selected_ds:
        target = DATASETS[ds_name]["target"]
        for cond in selected_cond:
            i += 1
            if (ds_name, cond) in done:
                print(f"[{i}/{total}] {ds_name} / {cond} — already done, skipping")
                continue

            print(f"\n[{i}/{total}] {ds_name} / {cond} — cleaning + fitting...")
            try:
                row = run_one(ds_name, cond, target)
                append_result(row)
                print(f"    F1={row['f1']:.4f}  AUC={row['auc']:.4f}  "
                      f"clean={row['cleaning_time_s']}s  fit={row['fit_time_s']}s  "
                      f"removed={row['rows_removed_label_noise']}")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")

    print(f"\nDone. Results in {RESULTS_FILE}")


if __name__ == "__main__":
    main()