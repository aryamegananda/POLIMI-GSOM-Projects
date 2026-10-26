import re
from pathlib import Path

import numpy as np
import pandas as pd


# 1. Folders and basic information

results_folder = Path("results")

pipelines = ["AutoML", "Hybrid", "Manual", "LLM"]
pipeline_names = {
    "automl": "AutoML",
    "hybrid": "Hybrid",
    "manual": "Manual",
    "llm": "LLM",
}

datasets = ["bank_marketing", "credit_card", "online_shoppers"]
target_columns = {
    "bank_marketing": "y",
    "credit_card": "default.payment.next.month",
    "online_shoppers": "Revenue",
}


# 2. Load experiment results

result_files = []
for pipeline in pipeline_names:
    file = results_folder / (pipeline + "_results.csv")
    result_files.append(pd.read_csv(file))

results = pd.concat(result_files, ignore_index=True)
results["pipeline"] = results["pipeline"].replace(pipeline_names)

# A failed run has no F1 score. Exclude it from performance averages.
successful = results[results["f1"].notna()].copy()

metric_columns = ["f1", "auc", "precision", "recall"]
summary = successful.groupby(["pipeline", "dataset", "condition"])[metric_columns].mean()
summary = summary.reset_index()


# 3. Predicted positive rate (Table 3.8)

# predicted positive rate = recall * actual positive rate / precision
actual_positive_rate = {}
for dataset in datasets:
    test_file = Path("data/processed") / dataset / "test.csv"
    test_data = pd.read_csv(test_file)
    actual_positive_rate[dataset] = test_data[target_columns[dataset]].mean()

print("Actual positive rate in the test sets:")
for dataset in datasets:
    print(f"  {dataset}: {actual_positive_rate[dataset]:.3f}")

successful["prevalence"] = successful["dataset"].map(actual_positive_rate)
successful["pred_pos_rate"] = (
    successful["recall"] * successful["prevalence"] / successful["precision"]
)

severe_noise = successful[successful["condition"] == "noise_severe"]
table_3_8 = severe_noise.groupby(["dataset", "pipeline"])[
    ["f1", "auc", "pred_pos_rate"]
].mean()
table_3_8 = table_3_8.unstack("pipeline")
table_3_8 = table_3_8.reindex(columns=pipelines, level=1)

print("\nSevere label noise (Table 3.8):")
print(table_3_8.round(3))
table_3_8.round(4).to_csv(results_folder / "table_severe_noise_behaviour.csv")


# 4. LLM tokens, API cost, and computation time 

# Historical experiment prices in USD per million tokens.
input_price = 2 / 1_000_000
output_price = 10 / 1_000_000

llm_runs = results[results["pipeline"] == "LLM"]
input_tokens = llm_runs["input_tokens"]
output_tokens = llm_runs["output_tokens"]

total_tokens = int(input_tokens.sum() + output_tokens.sum())
api_cost = input_tokens.sum() * input_price + output_tokens.sum() * output_price
failed_runs = llm_runs["f1"].isna().sum()

print(f"\nLLM runs: {len(llm_runs)}, failed: {failed_runs}")
print(f"Median input tokens: {input_tokens.median():.0f}")
print(f"Median output tokens: {output_tokens.median():.0f}")
print(f"Mean total tokens per run: {total_tokens / len(llm_runs):.0f}")
print(f"Total tokens: {total_tokens:,}; API cost: ${api_cost:.2f}")

time_columns = [
    "cleaning_time_s", "fit_time_s", "predict_time_s",
    "exec_time_s", "generation_time_s",
]

available_time_columns = []
for column in time_columns:
    if column in results.columns:
        available_time_columns.append(column)

results["total_time_s"] = results[available_time_columns].fillna(0).sum(axis=1)
median_time = results.groupby("pipeline")["total_time_s"].median()
median_time = median_time[pipelines]

print("\nMedian computation time per run (seconds):")
print(median_time.round(1))


# 5. Reproducibility (Table 3.12)

max_run_difference = {}

for pipeline in ["AutoML", "Hybrid"]:
    pipeline_runs = successful[successful["pipeline"] == pipeline]
    paired_runs = pipeline_runs.pivot_table(
        index=["dataset", "condition"], columns="run", values="f1"
    )

    differences = (paired_runs[1] - paired_runs[2]).abs()
    max_run_difference[pipeline] = differences.max()

    exact_matches = (differences == 0).sum()
    rounded_matches = (differences.round(4) == 0).sum()
    print(f"\n{pipeline}: exact matches {exact_matches}/21")
    print(f"  Matches to 4 decimals: {rounded_matches}/21")
    print(f"  Mean difference: {differences.mean():.4f}")
    print(f"  Largest difference: {differences.max():.4f}")

# For LLM, compare the best and worst successful runs for each condition.
llm_successful = successful[successful["pipeline"] == "LLM"]
llm_groups = llm_successful.groupby(["dataset", "condition"])["f1"]
llm_ranges = llm_groups.max() - llm_groups.min()
max_run_difference["LLM"] = llm_ranges.max()

print(f"\nLLM mean best-minus-worst F1: {llm_ranges.mean():.4f}")
print(f"LLM largest best-minus-worst F1: {llm_ranges.max():.4f}")
print(f"Condition with largest difference: {llm_ranges.idxmax()}")
llm_ranges.round(4).rename("f1_best_minus_worst").to_csv(
    results_folder / "table_llm_run_range.csv"
)

# Compare the Manual runs if the rerun CSV is available.
rerun_file = results_folder / "manual_results_rerun.csv"
if rerun_file.exists():
    manual_original = pd.read_csv(results_folder / "manual_results.csv")
    manual_rerun = pd.read_csv(rerun_file)

    manual_original = manual_original.set_index(["dataset", "condition"])
    manual_rerun = manual_rerun.set_index(["dataset", "condition"])
    compare_columns = ["f1", "auc", "best_model", "decision_threshold"]

    matching_rows = (
        manual_original[compare_columns]
        == manual_rerun.loc[manual_original.index, compare_columns]
    ).all(axis=1)
    manual_identical = bool(matching_rows.all())
    max_run_difference["Manual"] = (
        manual_original["f1"] - manual_rerun.loc[manual_original.index, "f1"]
    ).abs().max()

    print(f"Manual identical reruns: {matching_rows.sum()}/{len(matching_rows)}")
else:
    # This preserves the thesis's REPORTED result, but it is not verified.
    manual_identical = True
    max_run_difference["Manual"] = 0.0
    print("WARNING: Manual rerun CSV was not found.")
    print("Using the reported identical result; this has NOT been verified.")


# 6. TOE scoring (Tables 3.13-3.15)

def higher_is_better(values):
    best = max(values.values())
    scores = {}
    for pipeline, value in values.items():
        scores[pipeline] = 1 + 4 * value / best
    return scores


def lower_is_better(values):
    best = min(values.values())
    scores = {}
    for pipeline, value in values.items():
        scores[pipeline] = 1 + 4 * best / value
    return scores


f1_table = summary.pivot_table(
    index=["dataset", "condition"], columns="pipeline", values="f1"
)
f1_table = f1_table[pipelines]

# Performance: average F1 across conditions except severe label noise.
not_severe = f1_table.index.get_level_values("condition") != "noise_severe"
performance = f1_table[not_severe].mean().round(3).to_dict()

# Robustness: loss in F1 between clean data and severe label noise.
clean_f1 = f1_table.xs("clean", level="condition")
severe_f1 = f1_table.xs("noise_severe", level="condition")
noise_loss = (clean_f1 - severe_f1).mean().round(3).to_dict()

# Hybrid excludes its 11 Oct changes (2.2 h): rerun setup + copy of the AutoML threshold fix (see 2.6.6)
development_hours = {"AutoML": 3.6, "Hybrid": 17.5, "Manual": 18.2, "LLM": 4.7}
hourly_rate = 16.5 

labour_cost = {}
for pipeline in pipelines:
    labour_cost[pipeline] = development_hours[pipeline] * hourly_rate

expertise_areas = {"AutoML": 1, "Hybrid": 2, "Manual": 3, "LLM": 1}
privacy_scores = {"AutoML": 5, "Hybrid": 5, "Manual": 5, "LLM": 2}

failure_count = {}
for pipeline in pipelines:
    pipeline_rows = results[results["pipeline"] == pipeline]
    failure_count[pipeline] = int(pipeline_rows["f1"].isna().sum())

reproducibility_scores = {}
for pipeline in pipelines:
    if pipeline == "Manual" and manual_identical:
        score = 5
    elif max_run_difference[pipeline] < 0.03 and failure_count[pipeline] == 0:
        score = 4
    else:
        score = 3
    reproducibility_scores[pipeline] = score

toe_scores = {
    "Performance, ordinary data (T)": higher_is_better(performance),
    "Robustness to label noise (T)": lower_is_better(noise_loss),
    "Computation time (O)": lower_is_better(median_time.round(1).to_dict()),
    "Reproducibility (T)": reproducibility_scores,
    "Development time (O)": lower_is_better(development_hours),
    "Cost (O)": lower_is_better(labour_cost),
    "Expertise required (O)": lower_is_better(expertise_areas),
    "Data privacy (E)": privacy_scores,
}
toe = pd.DataFrame(toe_scores).T[pipelines]

raw_values = {
    "mean F1 excl. severe noise": performance,
    "mean F1 loss, severe noise": noise_loss,
    "median time per run (s)": median_time.to_dict(),
    "max run difference": max_run_difference,
    "development hours": development_hours,
    "labour cost (EUR)": labour_cost,
    "knowledge areas": expertise_areas,
}
raw = pd.DataFrame(raw_values).T[pipelines]

print("\nTOE raw values:")
print(raw.round(4))
print("\nTOE scores (5 = best):")
print(toe.round(2))
toe.round(2).to_csv(results_folder / "table_toe_scores.csv")
raw.round(4).to_csv(results_folder / "table_toe_raw_values.csv")


# 9. Keyword audit of generated LLM scripts

scripts = sorted(Path("transcripts").glob("*/*/run*/script.py"))

label_pattern = re.compile(
    r"cleanlab|label_issue|mislabel|noisy.?label|relabel|label.?noise", re.I
)
threshold_pattern = re.compile(r"threshold|thresh|best_t\b|cutoff|>=\s*t\b", re.I)
pdays_pattern = re.compile(
    r"pdays.*(never|contact)|(never|contact).*pdays|999.*(never|contact)", re.I
)

label_hits = []
threshold_hits = []
pdays_hits = []
bank_scripts = []

for script in scripts:
    text = script.read_text(encoding="utf-8")

    if label_pattern.search(text):
        label_hits.append(script)
    if threshold_pattern.search(text):
        threshold_hits.append(script)

    if script.parts[1] == "bank_marketing":
        bank_scripts.append(script)
        if pdays_pattern.search(text):
            pdays_hits.append(script)

print(f"\nLLM scripts found: {len(scripts)}")
print(f"Scripts mentioning label checking/cleaning: {len(label_hits)}")
print(f"Scripts mentioning a decision threshold: {len(threshold_hits)}")
print(f"Bank scripts mentioning the pdays=999 interpretation: "
      f"{len(pdays_hits)}/{len(bank_scripts)}")

for script in pdays_hits:
    print("  ", "/".join(script.parts[1:4]))
