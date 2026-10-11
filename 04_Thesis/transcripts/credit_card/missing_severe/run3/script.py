import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
TARGET = "default.payment.next.month"

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train[TARGET].values
Xtr = train.drop(columns=[TARGET])
Xte = test[Xtr.columns]

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAMT = [f"PAY_AMT{i}" for i in range(1, 7)]


def fe(df):
    d = df.copy()
    P = d[PAY].values
    B = d[BILL].values
    A = d[PAMT].values
    d["n_missing"] = d.isna().sum(axis=1)
    d["pay_mean"] = np.nanmean(P, axis=1)
    d["pay_max"] = np.nanmax(P, axis=1)
    d["pay_min"] = np.nanmin(P, axis=1)
    d["pay_late_cnt"] = np.nansum(P > 0, axis=1)
    d["pay_sum_pos"] = np.nansum(np.where(P > 0, P, 0), axis=1)
    d["pay_recent"] = np.nanmean(P[:, :3], axis=1)
    d["bill_mean"] = np.nanmean(B, axis=1)
    d["bill_max"] = np.nanmax(B, axis=1)
    d["bill_std"] = np.nanstd(B, axis=1)
    d["payamt_mean"] = np.nanmean(A, axis=1)
    d["payamt_sum"] = np.nansum(A, axis=1)
    d["payamt_max"] = np.nanmax(A, axis=1)
    d["util_mean"] = d["bill_mean"] / d["LIMIT_BAL"]
    d["util1"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    d["pay_to_bill"] = d["payamt_mean"] / (np.abs(d["bill_mean"]) + 1)
    for i in range(1, 6):
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (np.abs(d[f"BILL_AMT{i+1}"]) + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["limit_minus_bill"] = d["LIMIT_BAL"] - d["bill_mean"]
    return d


Xtr = fe(Xtr)
Xte = fe(Xte)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(len(Xtr))
oof_c = np.zeros(len(Xtr))
te_l = np.zeros(len(Xte))
te_c = np.zeros(len(Xte))

for tr, va in skf.split(Xtr, y):
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
        random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xtr.iloc[tr], y[tr], eval_set=[(Xtr.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
    te_l += m.predict_proba(Xte)[:, 1] / 5

    c = CatBoostClassifier(
        iterations=2000, learning_rate=0.04, depth=6, l2_leaf_reg=5,
        random_seed=SEED, verbose=0, early_stopping_rounds=150,
        eval_metric="Logloss", thread_count=-1)
    c.fit(Xtr.iloc[tr], y[tr], eval_set=(Xtr.iloc[va], y[va]))
    oof_c[va] = c.predict_proba(Xtr.iloc[va])[:, 1]
    te_c += c.predict_proba(Xte)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.3, -1
for t in np.arange(0.15, 0.6, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "OOF F1", best_f)

out = pd.DataFrame({"proba": np.clip(te, 0, 1), "label": (te >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
