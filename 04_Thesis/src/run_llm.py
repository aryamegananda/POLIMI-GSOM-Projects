# 0. Import
import os
import re
import sys
import time
import shutil
import tempfile
import platform
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# 1. Config
PIPELINE = "llm"
MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000
N_RUNS = 3
TIME_LIMIT_S = 900
RESULTS_FILE = "results/llm_results.csv"
TRANSCRIPTS_DIR = "transcripts"

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

PROMPT_TEMPLATE = """You are a data scientist. Build a complete machine learning pipeline for a
binary classification task.

## Task
Predict the target column "{target}" (1 = positive class, 0 = negative class).
Your pipeline will be evaluated on the F1 score of the positive class on a
held-out test set, so optimize for F1.

## Files (in the current working directory)
- train.csv          : training data, including the target column "{target}"
- test_features.csv  : test data, same columns but WITHOUT the target

## Data profile (training data)
{data_profile}

## Your job
Write ONE complete Python script that:
1. Loads train.csv and test_features.csv
2. Prepares the data as you see fit
3. Trains a model on the training data
4. Predicts the test rows and saves predictions.csv with exactly these columns:
   - "proba": predicted probability of the positive class (between 0 and 1)
   - "label": predicted class (0 or 1)
   in the same row order as test_features.csv

## Constraints
- Python 3, using only: pandas, numpy, scikit-learn, scipy, lightgbm, xgboost, catboost
- Do not use AutoML frameworks
- No internet access; do not download anything
- Use random_state=42 wherever randomness is involved
- The script must finish within 15 minutes on a laptop CPU
- This is a single attempt: you will not see any output or errors

Return only the Python script, in a single ```python code block.
"""


# 2. Data profile
def build_profile(train, target):
    lines = []
    lines.append(f"Rows: {len(train)}, Columns: {train.shape[1]}")

    lines.append("\nColumns (type, missing %):")
    for col in train.columns:
        missing = train[col].isnull().mean() * 100
        lines.append(f"- {col}: {train[col].dtype}, {missing:.1f}% missing")

    num_cols = train.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        lines.append("\nNumeric summary:")
        lines.append(train[num_cols].describe().T.round(3).to_string())

    cat_cols = train.select_dtypes(exclude="number").columns
    if len(cat_cols) > 0:
        lines.append("\nText/boolean columns (most common values):")
        for col in cat_cols:
            top = train[col].value_counts().head(5)
            values = ", ".join(f"{k} ({v})" for k, v in top.items())
            lines.append(f"- {col}: {values}")

    lines.append("\nTarget class balance:")
    balance = train[target].value_counts(normalize=True).sort_index() * 100
    for k, v in balance.items():
        lines.append(f"- {k}: {v:.1f}%")

    lines.append("\nFirst 5 rows:")
    lines.append(train.head(5).to_string())

    return "\n".join(lines)


# 3. Helpers
def extract_code(text):
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match is None:
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match is None:
        return None
    return match.group(1)


def save_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text is not None else "")


def run_script(code, work_dir):
    script_path = os.path.join(work_dir, "script.py")
    save_text(script_path, code)

    # the generated script must not see the API key
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "script.py"],
            cwd=work_dir, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=TIME_LIMIT_S,
        )
        status = "ok" if result.returncode == 0 else "error"
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = e.stderr if isinstance(e.stderr, str) else ""
    exec_time = time.perf_counter() - t0

    return status, stdout, stderr, exec_time


def score_predictions(pred_path, y_test):
    # returns (metrics dict, None) or (None, reason)
    if not os.path.exists(pred_path):
        return None, "predictions.csv not created"
    pred = pd.read_csv(pred_path)
    if "proba" not in pred.columns or "label" not in pred.columns:
        return None, f"wrong columns: {list(pred.columns)}"
    if len(pred) != len(y_test):
        return None, f"wrong number of rows: {len(pred)} instead of {len(y_test)}"
    if pred["proba"].isnull().any() or pred["label"].isnull().any():
        return None, "missing values in predictions"

    proba = pred["proba"].astype(float).values
    label = pred["label"].astype(int).values
    if not set(np.unique(label)) <= {0, 1}:
        return None, f"labels are not 0/1: {sorted(set(np.unique(label)))[:5]}"

    metrics = {
        "f1": f1_score(y_test, label, zero_division=0),
        "auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, label, zero_division=0),
        "recall": recall_score(y_test, label, zero_division=0),
    }
    return metrics, None


# 4. One run
def run_one(client, ds_name, cond, run, target):
    train = pd.read_csv(f"data/messy/{ds_name}/{cond}/train.csv")
    test = pd.read_csv(f"data/processed/{ds_name}/test.csv")
    y_test = test[target].values
    test_features = test.drop(columns=[target])

    log_dir = os.path.join(TRANSCRIPTS_DIR, ds_name, cond, f"run{run}")

    # temporary folder OUTSIDE the project, neutral file names
    work_dir = tempfile.mkdtemp(prefix="llm_run_")
    try:
        train.to_csv(os.path.join(work_dir, "train.csv"), index=False)
        test_features.to_csv(os.path.join(work_dir, "test_features.csv"), index=False)

        # Prompt
        prompt = PROMPT_TEMPLATE.format(target=target, data_profile=build_profile(train, target))
        save_text(os.path.join(log_dir, "prompt.txt"), prompt)

        # Call Claude (single-shot)
        t0 = time.perf_counter()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        generation_time = time.perf_counter() - t0

        answer = "".join(block.text for block in response.content if block.type == "text")
        save_text(os.path.join(log_dir, "response.txt"), answer)

        row = {
            "pipeline": PIPELINE,
            "dataset": ds_name,
            "condition": cond,
            "run": run,
            "f1": np.nan, "auc": np.nan, "precision": np.nan, "recall": np.nan,
            "status": None,
            "error": "",
            "generation_time_s": round(generation_time, 1),
            "exec_time_s": np.nan,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason,
            "n_train": len(train),
            "n_test": len(test),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        # Extract code
        code = extract_code(answer)
        if code is None:
            row["status"] = "no_code"
            row["error"] = "no python code block in the answer"
            return row
        save_text(os.path.join(log_dir, "script.py"), code)

        # Run the generated script
        status, stdout, stderr, exec_time = run_script(code, work_dir)
        save_text(os.path.join(log_dir, "stdout.txt"), stdout)
        save_text(os.path.join(log_dir, "stderr.txt"), stderr)
        row["exec_time_s"] = round(exec_time, 1)

        if status != "ok":
            row["status"] = status
            last_lines = [l for l in (stderr or "").strip().splitlines() if l.strip()]
            row["error"] = last_lines[-1][:300] if last_lines else ""
            return row

        # Score
        metrics, problem = score_predictions(os.path.join(work_dir, "predictions.csv"), y_test)
        if metrics is None:
            row["status"] = "bad_output"
            row["error"] = problem
            return row

        row.update(metrics)
        row["status"] = "ok"
        return row

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# 5. Saving + main
def already_done():
    done = []
    if os.path.exists(RESULTS_FILE):
        results = pd.read_csv(RESULTS_FILE)
        for _, r in results.iterrows():
            done.append((r["dataset"], r["condition"], int(r["run"])))
    return done


def save_result(row):
    os.makedirs("results", exist_ok=True)
    first_time = not os.path.exists(RESULTS_FILE)
    pd.DataFrame([row]).to_csv(RESULTS_FILE, mode="a", index=False, header=first_time)


def main():
    import anthropic
    load_dotenv()
    client = anthropic.Anthropic()

    datasets = list(DATASETS)
    conditions = CONDITIONS
    if len(sys.argv) >= 2:
        datasets = [sys.argv[1]]
    if len(sys.argv) >= 3:
        conditions = [sys.argv[2]]

    print(f"Model {MODEL} | Python {platform.python_version()} | "
          f"{platform.system()} {platform.release()} | {platform.processor()}")

    done = already_done()

    for ds_name in datasets:
        target = DATASETS[ds_name]["target"]
        for cond in conditions:
            for run in range(1, N_RUNS + 1):
                if (ds_name, cond, run) in done:
                    print(f"{ds_name} / {cond} / run {run}: already done, skipping")
                    continue

                print(f"\n{ds_name} / {cond} / run {run}")
                try:
                    row = run_one(client, ds_name, cond, run, target)
                except Exception as e:
                    # API / connection problems: not saved, retried next time
                    print(f"    API ERROR (not saved, will retry next time): {e}")
                    continue

                save_result(row)
                if row["status"] == "ok":
                    print(f"    ok  F1={row['f1']:.4f}  AUC={row['auc']:.4f}  "
                          f"gen={row['generation_time_s']}s  exec={row['exec_time_s']}s")
                else:
                    print(f"    FAILED ({row['status']}): {row['error']}")


if __name__ == "__main__":
    main()
