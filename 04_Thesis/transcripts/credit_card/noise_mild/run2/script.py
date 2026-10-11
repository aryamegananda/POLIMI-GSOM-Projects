import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import lightgbm as lgb

TARGET = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

def fe(df):
    d = df.copy()
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pa = [f"PAY_AMT{i}" for i in range(1, 7)]
    d["pay_mean"] = d[pays].mean(axis=1)
    d["pay_max"] = d[pays].max(axis=1)
    d["pay_sum_pos"] = d[pays].clip(lower=0).sum(axis=1)
    d["n_late"] = (d[pays] > 0).sum(axis=1)
    d["bill_mean"] = d[bills].mean(axis=1)
    d["payamt_mean"] = d[pa].mean(axis=1)
    d["payamt_sum"] = d[pa].sum(axis=1)
    d["bill_sum"] = d[bills].sum(axis=1)
    d["pay_ratio"] = d["payamt_sum"] / (d["bill_sum"].abs() + 1)
    d["util1"] = d["BILL_AMT1"] / (d["LIMIT_BAL"] + 1)
    d["util_mean"] = d["bill_mean"] / (d["LIMIT_BAL"] + 1)
    for i in range(1, 6):
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i+1}"].abs() + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["pay_trend"] = d["PAY_0"] - d["PAY_6"]
    return d

X = fe(train.drop(columns=[TARGET]))
y = train[TARGET].values
Xt = fe(test)[X.columns]

params = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
              random_state=42, verbose=-1, n_jobs=-1)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X)); oof_g = np.zeros(len(X))
pt_l = np.zeros(len(Xt)); pt_g = np.zeros(len(Xt))

def lin_prep(df):
    d = df.copy()
    for c in d.columns:
        if "BILL" in c or "PAY_AMT" in c or c in ("LIMIT_BAL", "payamt_sum", "bill_sum", "bill_mean", "payamt_mean", "bill_trend"):
            d[c] = np.sign(d[c]) * np.log1p(np.abs(d[c]))
    d["pay_ratio"] = d["pay_ratio"].clip(0, 10)
    for i in range(1, 6):
        d[f"ratio{i}"] = d[f"ratio{i}"].clip(0, 10)
    d["util1"] = d["util1"].clip(-2, 5)
    d["util_mean"] = d["util_mean"].clip(-2, 5)
    return d

XL = lin_prep(X); XtL = lin_prep(Xt)

for tr, va in skf.split(X, y):
    g = lgb.LGBMClassifier(**params)
    g.fit(X.iloc[tr], y[tr])
    oof_g[va] = g.predict_proba(X.iloc[va])[:, 1]
    pt_g += g.predict_proba(Xt)[:, 1] / 5
    l = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=42))
    l.fit(XL.iloc[tr], y[tr])
    oof_l[va] = l.predict_proba(XL.iloc[va])[:, 1]
    pt_l += l.predict_proba(XtL)[:, 1] / 5

best = (0, 0, 0.3)
for w in [1.0, 0.8, 0.7, 0.5]:
    oof = w * oof_g + (1 - w) * oof_l
    for t in np.arange(0.15, 0.6, 0.01):
        f = f1_score(y, oof >= t)
        if f > best[0]:
            best = (f, w, t)
f, w, t = best
print("CV F1", f, "w", w, "thr", t)

proba = np.clip(w * pt_g + (1 - w) * pt_l, 0, 1)
pd.DataFrame({"proba": proba, "label": (proba >= t).astype(int)}).to_csv("predictions.csv", index=False)
