import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

TARGET = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

feats = [c for c in test.columns if c != TARGET]
pay = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
pamt = [f"PAY_AMT{i}" for i in range(1, 7)]


def fe(df):
    d = df[feats].copy()
    d = d.drop(columns=["ID"])
    d["PAY_MAX"] = d[pay].max(axis=1)
    d["PAY_MEAN"] = d[pay].mean(axis=1)
    d["PAY_SUM"] = d[pay].sum(axis=1)
    d["PAY_POS_CNT"] = (d[pay] > 0).sum(axis=1)
    d["BILL_MEAN"] = d[bill].mean(axis=1)
    d["PAYAMT_MEAN"] = d[pamt].mean(axis=1)
    d["PAYAMT_SUM"] = d[pamt].sum(axis=1)
    d["UTIL1"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    d["UTIL_MEAN"] = d["BILL_MEAN"] / d["LIMIT_BAL"]
    d["PAY_RATIO"] = d["PAYAMT_SUM"] / (d[bill].sum(axis=1).abs() + 1)
    for i in range(1, 6):
        d[f"RATIO{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i+1}"].abs() + 1)
    d["BILL_TREND"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["LIMIT_PAYAMT"] = d["PAYAMT_MEAN"] / d["LIMIT_BAL"]
    return d.replace([np.inf, -np.inf], np.nan)


# drop training rows with missing target? target has none missing
X = fe(train)
y = train[TARGET].values
Xt = fe(test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X))
oof_g = np.zeros(len(X))
pt_l = np.zeros(len(Xt))
pt_g = np.zeros(len(Xt))

# clipped version for logistic regression
def lin_prep(d):
    d = d.copy()
    for c in d.columns:
        if c.startswith(("BILL", "PAY_AMT", "PAYAMT", "LIMIT_PAYAMT", "RATIO", "UTIL", "PAY_RATIO")):
            d[c] = np.sign(d[c]) * np.log1p(np.abs(d[c]))
    return d

XL, XtL = lin_prep(X), lin_prep(Xt)

for tr, va in skf.split(X, y):
    lg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.5, max_iter=2000, random_state=42))
    lg.fit(XL.iloc[tr], y[tr])
    oof_l[va] = lg.predict_proba(XL.iloc[va])[:, 1]
    pt_l += lg.predict_proba(XtL)[:, 1] / 5

    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=5, random_state=42,
                           verbose=-1, n_jobs=-1)
    m.fit(X.iloc[tr], y[tr])
    oof_g[va] = m.predict_proba(X.iloc[va])[:, 1]
    pt_g += m.predict_proba(Xt)[:, 1] / 5

best = (0, 0.5, 0.5)
for w in [0.0, 0.2, 0.3, 0.5]:
    p = w * oof_l + (1 - w) * oof_g
    for t in np.arange(0.15, 0.7, 0.01):
        f = f1_score(y, (p > t).astype(int))
        if f > best[0]:
            best = (f, w, t)
f, w, t = best
print("CV F1", f, "w", w, "thr", t)

proba = w * pt_l + (1 - w) * pt_g
label = (proba > t).astype(int)
pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
