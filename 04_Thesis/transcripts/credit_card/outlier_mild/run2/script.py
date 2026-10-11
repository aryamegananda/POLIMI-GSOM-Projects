import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

SEED = 42
TARGET = "default.payment.next.month"

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")


def fe(df):
    d = df.copy()
    # clean noisy values
    d["AGE"] = d["AGE"].where(d["AGE"] > 0, np.nan)
    d["LIMIT_BAL"] = d["LIMIT_BAL"].where(d["LIMIT_BAL"] > 0, np.nan)
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    amts = [f"PAY_AMT{i}" for i in range(1, 7)]
    d["pay_mean"] = d[pays].mean(axis=1)
    d["pay_max"] = d[pays].max(axis=1)
    d["pay_sum_pos"] = d[pays].clip(lower=0).sum(axis=1)
    d["n_late"] = (d[pays] > 0).sum(axis=1)
    d["bill_mean"] = d[bills].mean(axis=1)
    d["pay_amt_mean"] = d[amts].mean(axis=1)
    d["pay_amt_sum"] = d[amts].sum(axis=1)
    d["util1"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    d["util_mean"] = d["bill_mean"] / d["LIMIT_BAL"]
    d["pay_ratio1"] = d["PAY_AMT1"] / (d["BILL_AMT2"].abs() + 1)
    d["pay_ratio_all"] = d["pay_amt_sum"] / (d[bills].abs().sum(axis=1) + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["unpaid1"] = d["BILL_AMT2"] - d["PAY_AMT1"]
    d["pay_trend"] = d["PAY_0"] - d["PAY_6"]
    return d


X = fe(train.drop(columns=[TARGET]))
y = train[TARGET].values
Xt = fe(test)[X.columns]

params = dict(
    n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
    random_state=SEED, verbose=-1, n_jobs=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
pred = np.zeros(len(Xt))
oof_lr = np.zeros(len(X))
pred_lr = np.zeros(len(Xt))

Xf = X.fillna(X.median())
Xtf = Xt.fillna(X.median())
for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[tr], y[tr])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    pred += m.predict_proba(Xt)[:, 1] / skf.n_splits

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    # robust scaling via clipping extreme values
    lr.fit(Xf.iloc[tr].clip(-1e6, 1e6), y[tr])
    oof_lr[va] = lr.predict_proba(Xf.iloc[va].clip(-1e6, 1e6))[:, 1]
    pred_lr += lr.predict_proba(Xtf.clip(-1e6, 1e6))[:, 1] / skf.n_splits

# blend
best_w, best_f, best_t = 1.0, -1, 0.5
for w in [1.0, 0.8, 0.6]:
    p = w * oof + (1 - w) * oof_lr
    for t in np.arange(0.15, 0.7, 0.01):
        f = f1_score(y, (p > t).astype(int))
        if f > best_f:
            best_f, best_w, best_t = f, w, t

proba = best_w * pred + (1 - best_w) * pred_lr
proba = np.clip(proba, 0, 1)
label = (proba > best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
