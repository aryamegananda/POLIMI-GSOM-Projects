import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

TARGET = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

pay = [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
pamt = [f"PAY_AMT{i}" for i in range(1, 7)]


def fe(df):
    d = df.drop(columns=[TARGET, "ID"], errors="ignore").copy()
    P = d[pay].values
    B = d[bill].values
    A = d[pamt].values
    d["pay_mean"] = np.nanmean(np.where(np.isnan(P), np.nan, P), axis=1)
    d["pay_max"] = np.nanmax(np.where(np.isnan(P), -99, P), axis=1)
    d["pay_max"] = d["pay_max"].where(d["pay_max"] > -99, np.nan)
    d["pay_delay_cnt"] = np.where(np.isnan(P), 0, P > 0).sum(axis=1)
    d["pay_sum_pos"] = np.nansum(np.clip(P, 0, None), axis=1)
    d["bill_mean"] = np.nanmean(B, axis=1)
    d["bill_max"] = np.nanmax(np.where(np.isnan(B), -1e12, B), axis=1)
    d["bill_max"] = d["bill_max"].where(d["bill_max"] > -1e12, np.nan)
    d["pamt_mean"] = np.nanmean(A, axis=1)
    d["pamt_sum"] = np.nansum(A, axis=1)
    d["util_mean"] = d["bill_mean"] / d["LIMIT_BAL"]
    for i in range(1, 7):
        d[f"util{i}"] = d[f"BILL_AMT{i}"] / d["LIMIT_BAL"]
    d["pay_ratio"] = d["pamt_mean"] / (d["bill_mean"].abs() + 1)
    for i in range(1, 6):
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i+1}"].abs() + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["pay_trend"] = d["PAY_0"] - d["PAY_6"]
    d["limit_per_age"] = d["LIMIT_BAL"] / d["AGE"]
    d = d.replace([np.inf, -np.inf], np.nan)
    return d


y = train[TARGET].values
X = fe(train)
Xt = fe(test)[X.columns]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X)); oof_c = np.zeros(len(X))
te_l = np.zeros(len(Xt)); te_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=5.0,
                           random_state=42, verbose=-1, n_jobs=-1)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(X.iloc[va])[:, 1]
    te_l += m.predict_proba(Xt)[:, 1] / 5

    c = CatBoostClassifier(iterations=1000, learning_rate=0.04, depth=5,
                           l2_leaf_reg=5, random_seed=42, verbose=0,
                           early_stopping_rounds=100, thread_count=-1)
    c.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
    oof_c[va] = c.predict_proba(X.iloc[va])[:, 1]
    te_c += c.predict_proba(Xt)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.35, -1
for t in np.arange(0.15, 0.71, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1", best_f, "threshold", best_t)

out = pd.DataFrame({"proba": np.clip(te, 0, 1),
                    "label": (te >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
