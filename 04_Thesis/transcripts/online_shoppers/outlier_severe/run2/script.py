import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

tr = pd.read_csv("train.csv")
te = pd.read_csv("test_features.csv")
y = tr["Revenue"].values
Xtr = tr.drop(columns=["Revenue"])
Xte = te.drop(columns=["Revenue"], errors="ignore")[Xtr.columns]

def prep(df):
    d = df.copy()
    months = ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in months:
        d["Month_"+m] = (d["Month"] == m).astype(int)
    for v in ["Returning_Visitor","New_Visitor","Other"]:
        d["Visitor_"+v] = (d["VisitorType"] == v).astype(int)
    d["Month_num"] = d["Month"].map({m:i for i,m in enumerate(
        ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])}).fillna(-1)
    d = d.drop(columns=["Month","VisitorType"])
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDur"] = d["Administrative_Duration"] + d["Informational_Duration"] + d["ProductRelated_Duration"]
    d["PV_pos"] = (d["PageValues"] > 0).astype(int)
    d["Exit_Bounce"] = d["ExitRates"] - d["BounceRates"]
    d["PV_x_Exit"] = d["PageValues"] * d["ExitRates"]
    d["Dur_per_page"] = d["ProductRelated_Duration"] / (np.abs(d["ProductRelated"]) + 1)
    return d

Ptr = prep(Xtr)
Pte = prep(Xte)[Ptr.columns]

def make_models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=2.0,
                                  random_state=42, n_jobs=4, verbose=-1),
        "cat": CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6,
                                  random_seed=42, verbose=0, thread_count=4),
        "xgb": xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                                 reg_lambda=2.0, random_state=42, n_jobs=4,
                                 eval_metric="logloss"),
    }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
names = list(make_models().keys())
oof = {n: np.zeros(len(Ptr)) for n in names}
for tri, vai in skf.split(Ptr, y):
    ms = make_models()
    for n, m in ms.items():
        m.fit(Ptr.iloc[tri], y[tri])
        oof[n][vai] = m.predict_proba(Ptr.iloc[vai])[:, 1]

blend_oof = np.mean([oof[n] for n in names], axis=0)
best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.75, 0.01):
    f = f1_score(y, (blend_oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t

ms = make_models()
preds = []
for n, m in ms.items():
    m.fit(Ptr, y)
    preds.append(m.predict_proba(Pte)[:, 1])
proba = np.mean(preds, axis=0)
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
