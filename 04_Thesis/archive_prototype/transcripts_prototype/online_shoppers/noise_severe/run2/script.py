import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
Xall = pd.concat([train.drop(columns=["Revenue"]), test], axis=0, ignore_index=True)

def fe(df):
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(int)
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDur"] = df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    df["DurPerPage"] = df["TotalDur"] / (df["TotalPages"] + 1)
    df["PRDurPerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["ExitMinusBounce"] = df["ExitRates"] - df["BounceRates"]
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["logPV"] = np.log1p(df["PageValues"])
    month_map = {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8,
                 "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df["MonthNum"] = df["Month"].map(month_map).fillna(0)
    df = pd.get_dummies(df, columns=["Month", "VisitorType"], dtype=int)
    return df

X = fe(Xall)
n = len(train)
Xtr = X.iloc[:n].reset_index(drop=True)
Xte = X.iloc[n:].reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = {k: np.zeros(n) for k in ["lgb", "xgb", "cat"]}
tp = {k: np.zeros(len(Xte)) for k in oof}

for tr, va in skf.split(Xtr, y):
    Xa, Xb = Xtr.iloc[tr], Xtr.iloc[va]
    ya, yb = y[tr], y[va]

    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                           min_child_samples=30, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5, random_state=42,
                           verbose=-1)
    m.fit(Xa, ya)
    oof["lgb"][va] = m.predict_proba(Xb)[:, 1]
    tp["lgb"] += m.predict_proba(Xte)[:, 1] / 5

    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.02, max_depth=4,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                          reg_lambda=5, random_state=42, n_jobs=4,
                          eval_metric="logloss")
    m.fit(Xa, ya)
    oof["xgb"][va] = m.predict_proba(Xb)[:, 1]
    tp["xgb"] += m.predict_proba(Xte)[:, 1] / 5

    m = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                           random_seed=42, verbose=0, thread_count=4)
    m.fit(Xa, ya)
    oof["cat"][va] = m.predict_proba(Xb)[:, 1]
    tp["cat"] += m.predict_proba(Xte)[:, 1] / 5

oof_b = np.mean([oof[k] for k in oof], axis=0)
test_b = np.mean([tp[k] for k in tp], axis=0)

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.8, 0.01):
    f = f1_score(y, (oof_b >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1:", best_f, "threshold:", best_t)

out = pd.DataFrame({"proba": np.clip(test_b, 0, 1),
                    "label": (test_b >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
