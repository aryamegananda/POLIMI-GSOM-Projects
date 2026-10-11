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
    d = df.drop(columns=[TARGET], errors="ignore").copy()
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pamt = [f"PAY_AMT{i}" for i in range(1, 7)]
    d["PAY_MAX"] = d[pays].max(axis=1)
    d["PAY_MEAN"] = d[pays].mean(axis=1)
    d["PAY_SUM_POS"] = d[pays].clip(lower=0).sum(axis=1)
    d["PAY_N_DELAY"] = (d[pays] > 0).sum(axis=1)
    d["BILL_MEAN"] = d[bills].mean(axis=1)
    d["PAYAMT_MEAN"] = d[pamt].mean(axis=1)
    d["PAYAMT_SUM"] = d[pamt].sum(axis=1)
    d["UTIL1"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    d["UTIL_MEAN"] = d["BILL_MEAN"] / d["LIMIT_BAL"]
    d["PAY_RATIO"] = d["PAYAMT_SUM"] / (d[bills].clip(lower=0).sum(axis=1) + 1)
    for i in range(1, 6):
        d[f"PR{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i+1}"].clip(lower=0) + 1)
    d["BILL_TREND"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["PAY_TREND"] = d["PAY_0"] - d["PAY_6"]
    return d


X = fe(train)
y = train[TARGET].values
Xt = fe(test)

params = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
              random_state=42, verbose=-1, n_jobs=-1)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
pt = np.zeros(len(Xt))
for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[tr], y[tr])
    p_lgb = m.predict_proba(X.iloc[va])[:, 1]
    pt_lgb = m.predict_proba(Xt)[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=42))
    Xs = np.sign(X) * np.log1p(np.abs(X))
    Xts = np.sign(Xt) * np.log1p(np.abs(Xt))
    lr.fit(Xs.iloc[tr], y[tr])
    p_lr = lr.predict_proba(Xs.iloc[va])[:, 1]
    pt_lr = lr.predict_proba(Xts)[:, 1]
    oof[va] = 0.8 * p_lgb + 0.2 * p_lr
    pt += (0.8 * pt_lgb + 0.2 * pt_lr) / 5

ths = np.arange(0.15, 0.7, 0.01)
f1s = [f1_score(y, (oof > t).astype(int)) for t in ths]
best = ths[int(np.argmax(f1s))]
print("best threshold", best, "oof F1", max(f1s))

out = pd.DataFrame({"proba": np.clip(pt, 0, 1), "label": (pt > best).astype(int)})
out.to_csv("predictions.csv", index=False)
