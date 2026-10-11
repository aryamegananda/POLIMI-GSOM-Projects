import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].values
Xtr = train.drop(columns=["Revenue"])
Xte = test.copy()

months = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def prep(df):
    d = df.copy()
    num = ["Administrative", "Administrative_Duration", "Informational",
           "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
           "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
    for c in num:
        d[c + "_abs"] = d[c].abs()
    d["PV_log"] = np.sign(d["PageValues"]) * np.log1p(d["PageValues"].abs())
    d["PV_pos"] = (d["PageValues"] > 0).astype(int)
    d["total_pages"] = d["Administrative_abs"] + d["Informational_abs"] + d["ProductRelated_abs"]
    d["total_dur"] = (d["Administrative_Duration_abs"] + d["Informational_Duration_abs"]
                      + d["ProductRelated_Duration_abs"])
    d["dur_per_page"] = d["total_dur"] / (d["total_pages"] + 1)
    d["pr_dur_per_page"] = d["ProductRelated_Duration_abs"] / (d["ProductRelated_abs"] + 1)
    d["exit_minus_bounce"] = d["ExitRates"] - d["BounceRates"]
    d["PV_x_exit"] = d["PageValues"] * d["ExitRates"]
    mo = d["Month"].astype(str)
    for m in sorted(set(mo.unique()) | set(months)):
        d["Month_" + m] = (mo == m).astype(int)
    vt = d["VisitorType"].astype(str)
    for v in ["Returning_Visitor", "New_Visitor", "Other"]:
        d["VT_" + v] = (vt == v).astype(int)
    d = d.drop(columns=["Month", "VisitorType"])
    return d


Ptr = prep(Xtr)
Pte = prep(Xte)
Pte = Pte.reindex(columns=Ptr.columns, fill_value=0)
Ptr = Ptr.astype(float)
Pte = Pte.astype(float)


def make_models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=2.0,
                                  random_state=42, verbose=-1, n_jobs=-1),
        "cat": CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6,
                                  random_seed=42, verbose=0, thread_count=-1),
        "xgb": xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                                 reg_lambda=2.0, random_state=42, n_jobs=-1,
                                 eval_metric="logloss", tree_method="hist"),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04,
                                              max_leaf_nodes=15, random_state=42),
    }


names = list(make_models().keys())
oof = {n: np.full(len(y), np.nan) for n in names}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in skf.split(Ptr, y):
    ms = make_models()
    for n, m in ms.items():
        try:
            m.fit(Ptr.iloc[tr], y[tr])
            oof[n][va] = m.predict_proba(Ptr.iloc[va])[:, 1]
        except Exception:
            pass

good = [n for n in names if not np.isnan(oof[n]).any()]
if not good:
    good = ["hgb"]
    oof["hgb"] = np.zeros(len(y))
oof_avg = np.mean([oof[n] for n in good], axis=0)

best_t, best_f = 0.5, -1
for t in np.arange(0.1, 0.8, 0.01):
    f = f1_score(y, (oof_avg >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t

ms = make_models()
preds = []
for n in good:
    try:
        ms[n].fit(Ptr, y)
        preds.append(ms[n].predict_proba(Pte)[:, 1])
    except Exception:
        pass
if not preds:
    m = HistGradientBoostingClassifier(random_state=42).fit(Ptr, y)
    preds.append(m.predict_proba(Pte)[:, 1])
proba = np.clip(np.mean(preds, axis=0), 0, 1)

out = pd.DataFrame({"proba": proba, "label": (proba >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
