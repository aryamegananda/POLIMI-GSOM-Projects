import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].values
cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]


def prep(df):
    d = df.drop(columns=["y"], errors="ignore").copy()
    d["age"] = d["age"].clip(17, 100)
    d["duration"] = d["duration"].clip(lower=0)
    d["campaign"] = d["campaign"].clip(lower=1)
    d["log_duration"] = np.log1p(d["duration"])
    d["was_contacted"] = (d["pdays"] != 999).astype(int)
    d["pdays_clean"] = d["pdays"].where(d["pdays"] != 999, -1)
    d["emp_euribor"] = d["euribor3m"] * d["nr.employed"]
    d["dur_per_campaign"] = d["duration"] / d["campaign"]
    return d


Xtr = prep(train)
Xte = prep(test)

# CatBoost frame (strings)
Xtr_cb = Xtr.copy()
Xte_cb = Xte.copy()
for c in cat_cols:
    Xtr_cb[c] = Xtr_cb[c].astype(str)
    Xte_cb[c] = Xte_cb[c].astype(str)

# LightGBM frame (integer codes with consistent mapping)
Xtr_lg = Xtr.copy()
Xte_lg = Xte.copy()
for c in cat_cols:
    cats = sorted(set(Xtr[c].astype(str)) | set(Xte[c].astype(str)))
    m = {v: i for i, v in enumerate(cats)}
    Xtr_lg[c] = Xtr[c].astype(str).map(m).astype(int)
    Xte_lg[c] = Xte[c].astype(str).map(m).astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lg = np.zeros(len(Xtr))
oof_cb = np.zeros(len(Xtr))
te_lg = np.zeros(len(Xte))
te_cb = np.zeros(len(Xte))

for tr_idx, va_idx in skf.split(Xtr, y):
    m1 = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=2.0, random_state=42, verbose=-1)
    m1.fit(Xtr_lg.iloc[tr_idx], y[tr_idx])
    oof_lg[va_idx] = m1.predict_proba(Xtr_lg.iloc[va_idx])[:, 1]
    te_lg += m1.predict_proba(Xte_lg)[:, 1] / 5

    m2 = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=42, verbose=0, thread_count=-1)
    m2.fit(Xtr_cb.iloc[tr_idx], y[tr_idx], cat_features=cat_cols)
    oof_cb[va_idx] = m2.predict_proba(Xtr_cb.iloc[va_idx])[:, 1]
    te_cb += m2.predict_proba(Xte_cb)[:, 1] / 5

oof = 0.5 * oof_lg + 0.5 * oof_cb
test_proba = 0.5 * te_lg + 0.5 * te_cb

best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.9, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("Best threshold", best_t, "OOF F1", best_f)

out = pd.DataFrame({
    "proba": np.clip(test_proba, 0, 1),
    "label": (test_proba >= best_t).astype(int),
})
out.to_csv("predictions.csv", index=False)
