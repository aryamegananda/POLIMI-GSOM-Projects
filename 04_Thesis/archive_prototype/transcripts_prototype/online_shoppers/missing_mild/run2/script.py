import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
X = train.drop(columns=["Revenue"])
Xt = test.copy()[X.columns]

num_cols = [c for c in X.columns if c not in ["Month", "VisitorType", "Weekend"]]
month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

def prep(df):
    d = df.copy()
    d["Weekend"] = d["Weekend"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["MonthNum"] = d["Month"].map(month_map).fillna(0)
    d["VisitorType"] = d["VisitorType"].astype(str)
    d["Month"] = d["Month"].astype(str)
    for c in num_cols:
        d[c + "_na"] = d[c].isna().astype(int) if c in ["PageValues", "ProductRelated"] else 0
    d = d.drop(columns=[c for c in d.columns if c.endswith("_na") and d[c].sum() == 0 and False])
    tot_pages = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    tot_dur = d["Administrative_Duration"] + d["Informational_Duration"] + d["ProductRelated_Duration"]
    d["TotPages"] = tot_pages
    d["TotDur"] = tot_dur
    d["DurPerPage"] = tot_dur / (tot_pages + 1)
    d["PRDurPerPage"] = d["ProductRelated_Duration"] / (d["ProductRelated"] + 1)
    d["LogPV"] = np.log1p(d["PageValues"])
    d["PV_x_Exit"] = d["PageValues"] * (1 - d["ExitRates"])
    return d

Xp, Xtp = prep(X), prep(Xt)

# Tree-model frame: categorical -> category codes
full = pd.concat([Xp, Xtp], axis=0, ignore_index=True)
for c in ["Month", "VisitorType"]:
    full[c] = full[c].astype("category").cat.codes
Xtr_t = full.iloc[:len(Xp)].reset_index(drop=True)
Xte_t = full.iloc[len(Xp):].reset_index(drop=True)

# Logistic frame
lin = pd.get_dummies(pd.concat([Xp, Xtp], axis=0, ignore_index=True),
                     columns=["Month", "VisitorType"], dtype=float)
lin = lin.fillna(lin.iloc[:len(Xp)].median())
for c in ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
          "PageValues", "TotDur", "Administrative", "Informational", "ProductRelated",
          "TotPages", "DurPerPage", "PRDurPerPage", "PV_x_Exit"]:
    lin[c] = np.log1p(lin[c].clip(lower=0))
mu, sd = lin.iloc[:len(Xp)].mean(), lin.iloc[:len(Xp)].std().replace(0, 1)
lin = (lin - mu) / sd
Xtr_l = lin.iloc[:len(Xp)].values
Xte_l = lin.iloc[len(Xp):].values

def lgb_model():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=5, random_state=SEED,
                              verbose=-1, n_jobs=4)

def cb_model():
    return CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                              random_seed=SEED, verbose=0, thread_count=4)

def lr_model():
    return LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros((len(y), 3))
pte = np.zeros((len(Xte_t), 3))
for tr, va in skf.split(Xtr_t, y):
    m = lgb_model().fit(Xtr_t.iloc[tr], y[tr])
    oof[va, 0] = m.predict_proba(Xtr_t.iloc[va])[:, 1]
    pte[:, 0] += m.predict_proba(Xte_t)[:, 1] / 5
    m = cb_model().fit(Xtr_t.iloc[tr], y[tr])
    oof[va, 1] = m.predict_proba(Xtr_t.iloc[va])[:, 1]
    pte[:, 1] += m.predict_proba(Xte_t)[:, 1] / 5
    m = lr_model().fit(Xtr_l[tr], y[tr])
    oof[va, 2] = m.predict_proba(Xtr_l[va])[:, 1]
    pte[:, 2] += m.predict_proba(Xte_l)[:, 1] / 5

w = np.array([0.4, 0.45, 0.15])
oof_p = oof @ w
te_p = pte @ w

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof_p >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "OOF F1", best_f)

out = pd.DataFrame({"proba": np.clip(te_p, 0, 1),
                    "label": (te_p >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
