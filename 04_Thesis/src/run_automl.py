# 0. Import
import os
import time
import shutil
import platform
from datetime import datetime

import pandas as pd
from importlib.metadata import version as pkg_version
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# 1. Config
SEED = 42
PIPELINE = "full_auto_automl"
RESULTS_FILE = "results/automl_results.csv"
MODELS_DIR = "models/automl"
DELETE_MODELS_AFTER_RUN = True   # best_quality models are large; set False to keep them

AG_FIT_ARGS = {
    "presets": "best_quality",
    "time_limit": 300,
    "dynamic_stacking": False,   # prevents time-budget overruns
}
AG_EVAL_METRIC = "f1"

# 2. Variable Dataset
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

    model_path = f"{MODELS_DIR}/{ds_name}/{cond}"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)  # never reuse a half-finished predictor

    # ── Fit ──
    t0 = time.perf_counter()
    predictor = TabularPredictor(
        label=target, eval_metric=AG_EVAL_METRIC, path=model_path, verbosity=1
    ).fit(train, **AG_FIT_ARGS)
    fit_time = time.perf_counter() - t0

    # ── Predict on clean test ──
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
        "fit_time_s": round(fit_time, 1),
        "predict_time_s": round(predict_time, 2),
        "best_model": best_model,
        "positive_class": predictor.positive_class,
        "decision_threshold": getattr(predictor, "decision_threshold", None),
        "n_train": len(train),
        "n_test": len(test),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if DELETE_MODELS_AFTER_RUN:
        shutil.rmtree(model_path, ignore_errors=True)

    return row


def main():
    import sys
    selected = sys.argv[1:] or list(DATASETS)   # e.g. python src/run_automl.py bank_marketing
    unknown = [d for d in selected if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Choose from {list(DATASETS)}")

    print(f"AutoGluon {pkg_version('autogluon.tabular')} | Python {platform.python_version()} | "
          f"{platform.system()} {platform.release()} | {platform.processor()}")

    done = already_done()
    total = len(selected) * len(CONDITIONS)
    i = 0

    for ds_name in selected:
        cfg = DATASETS[ds_name]
        for cond in CONDITIONS:
            i += 1
            if (ds_name, cond) in done:
                print(f"[{i}/{total}] {ds_name} / {cond} — already done, skipping")
                continue

            print(f"\n[{i}/{total}] {ds_name} / {cond} — fitting...")
            try:
                row = run_one(ds_name, cond, cfg["target"])
                append_result(row)
                print(f"    F1={row['f1']:.4f}  AUC={row['auc']:.4f}  "
                      f"fit={row['fit_time_s']}s  best={row['best_model']}")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")

    print(f"\nDone. Results in {RESULTS_FILE}")


if __name__ == "__main__":
    main()
