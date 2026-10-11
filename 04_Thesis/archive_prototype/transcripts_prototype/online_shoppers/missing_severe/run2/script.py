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
train_X = train.drop(columns=["Revenue"])
test_X = test.copy()

num_cols = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
            "SpecialDay", "OperatingSystems", "Browser", "Region", "TrafficType"]

month_order = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6, "Jul": 7,
               "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

all_X = pd.concat([train_X, test_X], axis=0, ignore_index=True)
months = sorted(all_X["Month"].astype(str).unique())
visitors = sorted(all_X["VisitorType"].astype(str).unique())


def prep(df):
    d = pd.DataFrame(index=df.index)
    for c in num_cols:
        d[c] = pd.to_numeric(df[c], errors="coerce")
    d["missing_flag"] = d["PageValues"].isna().astype(int)
    d["Weekend"] = df["Weekend"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["month_num"] = df["Month"].astype(str).map(month_order)
    for m in months:
        d["month_" + m] = (df["Month"].astype(str) == m).astype(int)
    for v in visitors:
        d["visitor_" + v] = (df["VisitorType"].astype(str) == v).astype(int)
    # engineered features
    d["total_pages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["total_duration"] = d["Administrative_Duration"] + d["Informational_Duration"] + d["ProductRelated_Duration"]
    d["avg_dur_product"] = d["ProductRelated_Duration"] / (d["ProductRelated"] + 1)
    d["avg_dur_admin"] = d["Administrative_Duration"] / (d["Administrative"] + 1)
    d["pv_log"] = np.log1p(d["PageValues"])
    d["exit_bounce_diff"] = d["ExitRates"] - d["BounceRates"]
    d["pv_x_exit"] = d["PageValues"] * d["ExitRates"]
    return d


Xtr = prep(train_X)
Xte = prep(test_X)[Xtr.columns]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_lgb = np.zeros(len(Xtr))
oof_cb = np.zeros(len(Xtr))
te_lgb = np.zeros(len(Xte))
te_cb = np.zeros(len(Xte))

for tr_idx, va_idx in skf.split(Xtr, y):
    Xa, Xb = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
    ya = y[tr_idx]

    m1 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=30,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                            reg_lambda=2.0, random_state=SEED, verbose=-1, n_jobs=-1)
    m1.fit(Xa, ya)
    oof_lgb[va_idx] = m1.predict_proba(Xb)[:, 1]
    te_lgb += m1.predict_proba(Xte)[:, 1] / skf.n_splits

    m2 = CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6, random_seed=SEED,
                            verbose=0, thread_count=-1)
    m2.fit(Xa, ya)
    oof_cb[va_idx] = m2.predict_proba(Xb)[:, 1]
    te_cb += m2.predict_proba(Xte)[:, 1] / skf.n_splits

oof = 0.5 * oof_lgb + 0.5 * oof_cb
te = 0.5 * te_lgb + 0.5 * te_cb

best_t, best_f = 0.5, -1
for t in np.arange(0.1, 0.9, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("Best threshold", best_t, "OOF F1", best_f)

pred = pd.DataFrame({"proba": np.clip(te, 0, 1), "label": (te >= best_t).astype(int)})
pred.to_csv("predictions.csv", index=False)
