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

y = train[TARGET].astype(int).values
Xtr = train.drop(columns=[TARGET])
Xte = test[Xtr.columns]

pay = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
pamt = [f"PAY_AMT{i}" for i in range(1, 7)]


def fe(df):
    d = df.copy()
    d["pay_mean"] = d[pay].mean(axis=1)
    d["pay_max"] = d[pay].max(axis=1)
    d["pay_min"] = d[pay].min(axis=1)
    d["pay_pos_cnt"] = (d[pay] > 0).sum(axis=1)
    d["pay_nan_cnt"] = d[pay].isna().sum(axis=1)
    d["bill_mean"] = d[bill].mean(axis=1)
    d["bill_max"] = d[bill].max(axis=1)
    d["pamt_mean"] = d[pamt].mean(axis=1)
    d["pamt_sum"] = d[pamt].sum(axis=1)
    d["util"] = d["bill_mean"] / (d["LIMIT_BAL"] + 1)
    d["util1"] = d["BILL_AMT1"] / (d["LIMIT_BAL"] + 1)
    d["pay_ratio"] = d["pamt_sum"] / (d[bill].sum(axis=1).abs() + 1)
    for i in range(1, 7):
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i}"].abs() + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["pay_trend"] = d["PAY_0"] - d["PAY_6"]
    return d


Xtr = fe(Xtr)
Xte = fe(Xte)

params = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
              random_state=42, verbose=-1, n_jobs=-1)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(Xtr))
pte = np.zeros(len(Xte))
for tr, va in skf.split(Xtr, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr.iloc[tr], y[tr])
    pl = m.predict_proba(Xtr.iloc[va])[:, 1]
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.5, max_iter=1000))
    lr.fit(Xtr.iloc[tr].replace([np.inf, -np.inf], np.nan), y[tr])
    pr = lr.predict_proba(Xtr.iloc[va].replace([np.inf, -np.inf], np.nan))[:, 1]
    oof[va] = 0.8 * pl + 0.2 * pr
    pte += (0.8 * m.predict_proba(Xte)[:, 1]
            + 0.2 * lr.predict_proba(Xte.replace([np.inf, -np.inf], np.nan))[:, 1]) / 5

ths = np.arange(0.15, 0.7, 0.01)
f1s = [f1_score(y, oof >= t) for t in ths]
best = ths[int(np.argmax(f1s))]
print("best threshold", best, "oof F1", max(f1s))

out = pd.DataFrame({"proba": np.clip(pte, 0, 1), "label": (pte >= best).astype(int)})
out.to_csv("predictions.csv", index=False)
