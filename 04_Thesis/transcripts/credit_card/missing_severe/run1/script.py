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

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT = [f"PAY_AMT{i}" for i in range(1, 7)]


def fe(df):
    d = df.copy()
    d["pay_mean"] = d[PAY].mean(axis=1)
    d["pay_max"] = d[PAY].max(axis=1)
    d["pay_min"] = d[PAY].min(axis=1)
    d["pay_late_cnt"] = (d[PAY] > 0).sum(axis=1)
    d["pay_sum_pos"] = d[PAY].clip(lower=0).sum(axis=1)
    d["bill_mean"] = d[BILL].mean(axis=1)
    d["bill_max"] = d[BILL].max(axis=1)
    d["amt_mean"] = d[AMT].mean(axis=1)
    d["amt_sum"] = d[AMT].sum(axis=1)
    d["amt_to_bill"] = d["amt_mean"] / (d["bill_mean"].abs() + 1)
    d["util_mean"] = d["bill_mean"] / (d["LIMIT_BAL"] + 1)
    for i in range(1, 7):
        d[f"util{i}"] = d[f"BILL_AMT{i}"] / (d["LIMIT_BAL"] + 1)
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i}"].abs() + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["bill_trend_mean"] = d[BILL[:3]].mean(axis=1) - d[BILL[3:]].mean(axis=1)
    d["pay_trend"] = d[PAY[:3]].mean(axis=1) - d[PAY[3:]].mean(axis=1)
    d["n_missing"] = df.isna().sum(axis=1)
    return d


y = train[TARGET].values
X = fe(train.drop(columns=[TARGET]))
Xt = fe(test[train.drop(columns=[TARGET]).columns])
X = X.replace([np.inf, -np.inf], np.nan)
Xt = Xt.replace([np.inf, -np.inf], np.nan)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
names = ["lgb", "xgb", "cat"]
oof = {n: np.zeros(len(X)) for n in names}
tp = {n: np.zeros(len(Xt)) for n in names}


def make(n):
    if n == "lgb":
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.6, reg_lambda=5.0,
                                  random_state=SEED, verbose=-1, n_jobs=-1)
    if n == "xgb":
        return xgb.XGBClassifier(n_estimators=350, learning_rate=0.03, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                                 reg_lambda=5.0, tree_method="hist",
                                 random_state=SEED, n_jobs=-1)
    return CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6,
                              l2_leaf_reg=5, random_seed=SEED, verbose=0,
                              thread_count=-1)


for tr, va in skf.split(X, y):
    for n in names:
        m = make(n)
        m.fit(X.iloc[tr], y[tr])
        oof[n][va] = m.predict_proba(X.iloc[va])[:, 1]
        tp[n] += m.predict_proba(Xt)[:, 1] / skf.n_splits

oof_blend = np.mean([oof[n] for n in names], axis=0)
test_blend = np.mean([tp[n] for n in names], axis=0)

best_t, best_f = 0.35, -1
for t in np.arange(0.15, 0.71, 0.005):
    f = f1_score(y, (oof_blend >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1:", best_f, "threshold:", best_t)

out = pd.DataFrame({
    "proba": np.clip(test_blend, 0, 1),
    "label": (test_blend >= best_t).astype(int),
})
out.to_csv("predictions.csv", index=False)
