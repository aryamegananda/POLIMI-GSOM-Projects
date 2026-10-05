import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
train_x = train.drop(columns=["Revenue"])
test_x = test.copy()
if "Revenue" in test_x.columns:
    test_x = test_x.drop(columns=["Revenue"])

n_train = len(train_x)
full = pd.concat([train_x, test_x], axis=0, ignore_index=True)

month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
full["Month_num"] = full["Month"].map(month_map)
full["Weekend"] = full["Weekend"].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)

num_cols = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
            "SpecialDay", "OperatingSystems", "Browser", "Region", "TrafficType"]
full["row_missing"] = full["PageValues"].isna().astype(int)

# engineered features (NaN propagates)
full["total_pages"] = full["Administrative"] + full["Informational"] + full["ProductRelated"]
full["total_duration"] = (full["Administrative_Duration"] + full["Informational_Duration"]
                          + full["ProductRelated_Duration"])
full["dur_per_product"] = full["ProductRelated_Duration"] / (full["ProductRelated"] + 1)
full["dur_per_page"] = full["total_duration"] / (full["total_pages"] + 1)
full["exit_bounce_diff"] = full["ExitRates"] - full["BounceRates"]
full["pv_x_exit"] = full["PageValues"] * (1 - full["ExitRates"])
full["log_pagevalues"] = np.log1p(full["PageValues"])

dummies = pd.get_dummies(full[["Month", "VisitorType"]].astype(str), dtype=int)
full = pd.concat([full.drop(columns=["Month", "VisitorType"]), dummies], axis=1)
full = full.replace([np.inf, -np.inf], np.nan)

X = full.iloc[:n_train].reset_index(drop=True)
Xt = full.iloc[n_train:].reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(n_train)
oof_c = np.zeros(n_train)
test_l = np.zeros(len(Xt))
test_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m1 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15,
                            min_child_samples=20, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.7, reg_lambda=2.0,
                            random_state=SEED, verbose=-1, n_jobs=4)
    m1.fit(X.iloc[tr], y[tr])
    oof_l[va] = m1.predict_proba(X.iloc[va])[:, 1]
    test_l += m1.predict_proba(Xt)[:, 1] / skf.n_splits

    m2 = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                            random_seed=SEED, verbose=0, thread_count=4)
    m2.fit(X.iloc[tr], y[tr])
    oof_c[va] = m2.predict_proba(X.iloc[va])[:, 1]
    test_c += m2.predict_proba(Xt)[:, 1] / skf.n_splits

oof = 0.5 * oof_l + 0.5 * oof_c
test_p = 0.5 * test_l + 0.5 * test_c

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.75, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1: %.4f at threshold %.2f" % (best_f, best_t))

out = pd.DataFrame({"proba": np.clip(test_p, 0, 1),
                    "label": (test_p >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
