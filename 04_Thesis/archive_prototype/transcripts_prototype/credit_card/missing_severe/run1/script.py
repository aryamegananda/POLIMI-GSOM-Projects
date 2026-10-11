import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
TARGET = "default.payment.next.month"

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")


def fe(df):
    df = df.copy()
    df = df.drop(columns=["ID"], errors="ignore")
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pamts = [f"PAY_AMT{i}" for i in range(1, 7)]
    df["pay_mean"] = df[pays].mean(axis=1, skipna=True)
    df["pay_max"] = df[pays].max(axis=1, skipna=True)
    df["pay_min"] = df[pays].min(axis=1, skipna=True)
    df["pay_delay_cnt"] = (df[pays] > 0).sum(axis=1) + df[pays].isna().all(axis=1) * np.nan
    df["bill_mean"] = df[bills].mean(axis=1, skipna=True)
    df["bill_max"] = df[bills].max(axis=1, skipna=True)
    df["pamt_sum"] = df[pamts].sum(axis=1, min_count=1)
    df["pamt_mean"] = df[pamts].mean(axis=1, skipna=True)
    df["util"] = df["bill_mean"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["util1"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["pay_ratio"] = df["pamt_sum"] / (df[bills].sum(axis=1, min_count=1).abs() + 1)
    for i in range(1, 6):
        df[f"ratio{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i+1}"].abs() + 1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["limit_minus_bill"] = df["LIMIT_BAL"] - df["BILL_AMT1"]
    return df


y = train[TARGET].values
X = fe(train.drop(columns=[TARGET]))
Xt = fe(test[[c for c in test.columns if c != TARGET]])
Xt = Xt[X.columns]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
names = ["lgb", "xgb", "cat"]
oof = {n: np.zeros(len(X)) for n in names}
tp = {n: np.zeros(len(Xt)) for n in names}

for tr, va in skf.split(X, y):
    Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y[tr], y[va]

    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5.0,
                           random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr)
    oof["lgb"][va] = m.predict_proba(Xva)[:, 1]
    tp["lgb"] += m.predict_proba(Xt)[:, 1] / 5

    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                          reg_lambda=5.0, random_state=SEED, n_jobs=-1,
                          tree_method="hist", eval_metric="logloss")
    m.fit(Xtr, ytr)
    oof["xgb"][va] = m.predict_proba(Xva)[:, 1]
    tp["xgb"] += m.predict_proba(Xt)[:, 1] / 5

    m = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                           random_seed=SEED, verbose=0, thread_count=-1)
    m.fit(Xtr, ytr)
    oof["cat"][va] = m.predict_proba(Xva)[:, 1]
    tp["cat"] += m.predict_proba(Xt)[:, 1] / 5

oof_ens = np.mean([oof[n] for n in names], axis=0)
test_ens = np.mean([tp[n] for n in names], axis=0)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof_ens >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1", best_f, "threshold", best_t)

out = pd.DataFrame({"proba": np.clip(test_ens, 0, 1),
                    "label": (test_ens >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
