import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

TARGET = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train[TARGET].values
Xtr = train.drop(columns=[TARGET])
Xte = test[Xtr.columns]

pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bills = [f"BILL_AMT{i}" for i in range(1, 7)]
pamts = [f"PAY_AMT{i}" for i in range(1, 7)]


def fe(df):
    d = df.copy()
    d["LIMIT_NEG"] = (d["LIMIT_BAL"] < 0).astype(int)
    d["AGE"] = d["AGE"].where(d["AGE"] >= 18, np.nan)
    d["LIMIT_BAL"] = d["LIMIT_BAL"].where(d["LIMIT_BAL"] > 0, np.nan)
    d["PAY_MAX"] = d[pays].max(axis=1)
    d["PAY_MIN"] = d[pays].min(axis=1)
    d["PAY_MEAN"] = d[pays].mean(axis=1)
    d["PAY_SUM"] = d[pays].sum(axis=1)
    d["PAY_POS_CNT"] = (d[pays] > 0).sum(axis=1)
    d["PAY_GE2_CNT"] = (d[pays] >= 2).sum(axis=1)
    d["PAY_REV_CNT"] = (d[pays] == -2).sum(axis=1)
    d["PAY_RECENT"] = d["PAY_0"] * 2 + d["PAY_2"]
    d["PAY_TREND"] = d["PAY_0"] - d["PAY_6"]
    d["BILL_SUM"] = d[bills].sum(axis=1)
    d["BILL_MEAN"] = d[bills].mean(axis=1)
    d["BILL_STD"] = d[bills].std(axis=1)
    d["PAYAMT_SUM"] = d[pamts].sum(axis=1)
    d["PAYAMT_MEAN"] = d[pamts].mean(axis=1)
    d["PAYAMT_STD"] = d[pamts].std(axis=1)
    lim = d["LIMIT_BAL"]
    for i in range(1, 7):
        d[f"UTIL{i}"] = d[f"BILL_AMT{i}"] / lim
    d["UTIL_MEAN"] = d["BILL_MEAN"] / lim
    d["PAYAMT_LIM"] = d["PAYAMT_MEAN"] / lim
    d["PAY_BILL_RATIO"] = d["PAYAMT_SUM"] / (d["BILL_SUM"].abs() + 1)
    for i in range(1, 6):
        d[f"PR{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i+1}"].abs() + 1)
    d["PR6"] = d["PAY_AMT6"] / (d["BILL_AMT6"].abs() + 1)
    d["BILL_DIFF"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["BILL_DIFF12"] = d["BILL_AMT1"] - d["BILL_AMT2"]
    d["UNPAID1"] = d["BILL_AMT2"] - d["PAY_AMT1"]
    d["UNPAID_MEAN"] = (d["BILL_AMT2"] - d["PAY_AMT1"] + d["BILL_AMT3"] - d["PAY_AMT2"]
                        + d["BILL_AMT4"] - d["PAY_AMT3"]) / 3
    d["UNPAID_LIM"] = d["UNPAID_MEAN"] / lim
    d = d.replace([np.inf, -np.inf], np.nan)
    return d


Xtr = fe(Xtr)
Xte = fe(Xte)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(Xtr)); te_l = np.zeros(len(Xte))
oof_c = np.zeros(len(Xtr)); te_c = np.zeros(len(Xte))

for tr, va in skf.split(Xtr, y):
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=5.0,
                           random_state=42, verbose=-1)
    m.fit(Xtr.iloc[tr], y[tr], eval_set=[(Xtr.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
    te_l += m.predict_proba(Xte)[:, 1] / 5

    c = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6,
                           l2_leaf_reg=5, random_seed=42, verbose=0,
                           early_stopping_rounds=100, eval_metric="Logloss")
    c.fit(Xtr.iloc[tr], y[tr], eval_set=(Xtr.iloc[va], y[va]))
    oof_c[va] = c.predict_proba(Xtr.iloc[va])[:, 1]
    te_c += c.predict_proba(Xte)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.3, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1", best_f, "threshold", best_t)

proba = np.clip(te, 0, 1)
pd.DataFrame({"proba": proba, "label": (proba >= best_t).astype(int)}).to_csv(
    "predictions.csv", index=False)
