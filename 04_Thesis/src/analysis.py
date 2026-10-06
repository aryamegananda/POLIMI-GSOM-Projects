# 0. Import
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


# 1. Load
pipelines = ["automl", "hybrid", "llm", "manual"]

dfs = {}
for p in pipelines:
    path = f"results/{p}_results.csv"
    dfs[p] = pd.read_csv(path)

results = pd.concat(dfs.values())


# 2. Averaging LLM result
names = {
    "full_auto_automl": "AutoML",
    "hybrid": "Hybrid",
    "manual": "Manual",
    "llm": "LLM"
}

results["pipeline"] = results["pipeline"].map(names)
metrics = ["f1", "auc", "precision", "recall"]
summary = results.groupby(["pipeline", "dataset", "condition"])[metrics].mean().reset_index()

print(summary.shape)
print(summary.head(10))


# 3. Main table
order = ["AutoML", "Hybrid", "Manual", "LLM"]

f1_table = summary.pivot_table(index=["dataset", "condition"], columns="pipeline", values="f1")
f1_table = f1_table[order].round(3)

f1_table.to_csv("results/table_f1.csv")

# 4. Friedman Test
stat, p_value = friedmanchisquare(
    f1_table["AutoML"], f1_table["Hybrid"], f1_table["Manual"], f1_table["LLM"]
)
print(f"Friedman chi-square = {stat:.3f}, p-value = {p_value:.4f}")

ranks = f1_table.rank(axis=1, ascending=False)
print(f"Average rank:")
print(ranks.mean().round(2))


# 5. Nemenyi post-hoc test
nemenyi = sp.posthoc_nemenyi_friedman(f1_table.values)
nemenyi.index = order
nemenyi.columns = order
print(f"\nNemenyi p-values (pairs with p<0.05 differ significantly):")
print(nemenyi.round(3))
nemenyi.round(4).to_csv("results/nemenyi_f1.csv")


# 6. Degradation effect: F1 change vs. each pipeline's own clean baseline
delta = summary.copy()
clean_f1 = summary[summary["condition"] == "clean"].set_index(["pipeline", "dataset"])["f1"]

delta_values = []
for _, row in delta.iterrows():
    baseline = clean_f1[(row["pipeline"], row["dataset"])]
    delta_values.append(row["f1"] - baseline)
delta["f1_change"] = delta_values

delta_table = delta.pivot_table(index=["dataset", "condition"], columns="pipeline", values="f1_change")
delta_table = delta_table[order]
print("\nF1 change vs. clean (negative = performance lost):")
print(delta_table.round(3))
delta_table.round(4).to_csv("results/table_f1_change.csv")


# 7. Average f1 change per condition
no_clean = delta[delta["condition"] != "clean"]
avg_change = no_clean.pivot_table(index="condition", columns="pipeline", values="f1_change", aggfunc="mean")
avg_change = avg_change[order]

print(f"\nAverage F1 change vs. clean (mean of 3 datasets):")
print(avg_change.round(3))
avg_change.round(4).to_csv("results/table_f1_change_avg.csv")


# 8. Computation time per pipeline
time_cols = ["cleaning_time_s", "fit_time_s", "predict_time_s", "exec_time_s"]
results["compute_time_s"] = 0
for col in time_cols:
    results["compute_time_s"] = results["compute_time_s"] + results[col].fillna(0)

time_table = results.groupby("pipeline")["compute_time_s"].agg(["median", "mean", "max"])
time_table = time_table.loc[order].round(1)
print("\nComputation time per run (seconds):")
print(time_table)
time_table.to_csv("results/table_compute_time.csv")

llm_gen = results[results["pipeline"] == "LLM"]["generation_time_s"]
print(f"\nLLM script generation time: median {llm_gen.median():.1f} s, max {llm_gen.max():.1f} s")


# 9. LLM reproducibility: F1 spread across the 3 runs of each condition
llm = results[results["pipeline"] == "LLM"]
llm_std = llm.groupby(["dataset", "condition"])["f1"].std().reset_index()
llm_std.columns = ["dataset", "condition", "f1_std"]

print("\nLLM F1 standard deviation across 3 runs:")
print(llm_std.round(4).to_string(index=False))
print(f"\nMean std (all conditions):      {llm_std['f1_std'].mean():.4f}")
print(f"Mean std (excl. noise_severe):  {llm_std[llm_std['condition'] != 'noise_severe']['f1_std'].mean():.4f}")
print(f"Mean std (noise_severe only):   {llm_std[llm_std['condition'] == 'noise_severe']['f1_std'].mean():.4f}")
llm_std.round(4).to_csv("results/table_llm_reproducibility.csv", index=False)