import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import lightgbm as lgb

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
Xtr = train.drop(columns=["Revenue"]).copy()
Xte = test.copy()
Xte = Xte[Xtr.columns]

months = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
mmap = {m: i for i, m in enumerate(months)}

num_cols = [c for c in Xtr.columns if c not in ["Month", "VisitorType", "Weekend"]]

def prep(df, ref):
    d = df.copy()
    d["Weekend"] = d["Weekend"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["MonthNum"] = d["Month"].map(mmap).astype(float)
    for c in num_cols:
        d[c + "_na"] = d[c].isna().astype(int) if c in ["PageValues", "Administrative"] else 0
    d["n_missing"] = d[num_cols].isna().sum(axis=1)
    for v in ["Returning_Visitor", "New_Visitor", "Other"]:
        d["VT_" + v] = (d["VisitorType"] == v).astype(int)
    for m in ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]:
        d["M_" + m] = (d["Month"] == m).astype(int)
    d = d.drop(columns=["Month", "VisitorType"])
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDur"] = d["Administrative_Duration"] + d["Informational_Duration"] + d["ProductRelated_Duration"]
    d["DurPerPage"] = d["ProductRelated_Duration"] / (d["ProductRelated"] + 1)
    d["PV_log"] = np.log1p(d["PageValues"])
    d["Exit_Bounce"] = d["ExitRates"] - d["BounceRates"]
    d["PV_x_Exit"] = d["PageValues"] * (1 - d["ExitRates"])
    return d

Xtr_p = prep(Xtr, Xtr)
Xte_p = prep(Xte, Xtr)
Xte_p = Xte_p[Xtr_p.columns]

# Models (NaN handled natively)
def make_lgb():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=5, random_state=SEED,
                              verbose=-1, n_jobs=4)

def make_hgb():
    return HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250, max_leaf_nodes=15,
                                          min_samples_leaf=30, l2_regularization=2.0,
                                          random_state=SEED)

from catboost import CatBoostClassifier
def make_cb():
    return CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                              random_seed=SEED, verbose=0, thread_count=4)

makers = [make_lgb, make_hgb, make_cb]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof = np.zeros(len(y))
for mk in makers:
    p = cross_val_predict(mk(), Xtr_p, y, cv=skf, method="predict_proba")[:, 1]
    oof += p / len(makers)

# Choose threshold maximizing OOF F1
best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

pte = np.zeros(len(Xte_p))
for mk in makers:
    m = mk()
    m.fit(Xtr_p, y)
    pte += m.predict_proba(Xte_p)[:, 1] / len(makers)

out = pd.DataFrame({"proba": np.clip(pte, 0, 1), "label": (pte >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
