import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

TARGET = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]


def fe(df):
    d = df.drop(columns=[c for c in ["ID", TARGET] if c in df.columns]).copy()
    d["anom"] = ((df["LIMIT_BAL"] < 0) | (df["AGE"] < 18) | (df["SEX"] == 3) |
                 (df["PAY_0"] == 9) | (df["EDUCATION"] == 7) | (df["MARRIAGE"] == 4)).astype(int)
    P = df[pay_cols]
    d["pay_sum"] = P.sum(axis=1)
    d["pay_max"] = P.max(axis=1)
    d["pay_mean"] = P.mean(axis=1)
    d["pay_pos_cnt"] = (P > 0).sum(axis=1)
    d["pay_neg_cnt"] = (P < 0).sum(axis=1)
    d["pay_recent"] = P["PAY_0"] + P["PAY_2"]
    d["bill_sum"] = df[bill_cols].sum(axis=1)
    d["bill_mean"] = df[bill_cols].mean(axis=1)
    d["amt_sum"] = df[amt_cols].sum(axis=1)
    d["amt_mean"] = df[amt_cols].mean(axis=1)
    lim = df["LIMIT_BAL"].where(df["LIMIT_BAL"] > 0, np.nan)
    d["util1"] = df["BILL_AMT1"] / lim
    d["util_mean"] = d["bill_mean"] / lim
    d["pay_ratio"] = d["amt_sum"] / (d["bill_sum"].abs() + 1)
    for i in range(1, 7):
        d[f"ratio{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"].abs() + 1)
    d["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    d["unpaid1"] = df["BILL_AMT1"] - df["PAY_AMT1"]
    d["unpaid_lim"] = d["unpaid1"] / lim
    d = d.replace([np.inf, -np.inf], np.nan)
    return d


X = fe(train)
y = train[TARGET].values
Xt = fe(test)[X.columns]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X)); oof_c = np.zeros(len(X))
te_l = np.zeros(len(Xt)); te_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=5, random_state=42,
                           n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(X.iloc[va])[:, 1]
    te_l += m.predict_proba(Xt)[:, 1] / 5

    c = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6,
                           random_seed=42, verbose=0, early_stopping_rounds=100,
                           thread_count=-1)
    c.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
    oof_c[va] = c.predict_proba(X.iloc[va])[:, 1]
    te_c += c.predict_proba(Xt)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.3, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof > t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1:", best_f, "threshold:", best_t)

out = pd.DataFrame({"proba": np.clip(te, 0, 1),
                    "label": (te > best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
