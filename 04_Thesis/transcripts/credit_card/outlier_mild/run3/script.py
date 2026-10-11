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
    df.loc[df["LIMIT_BAL"] <= 0, "LIMIT_BAL"] = np.nan
    df.loc[df["AGE"] < 18, "AGE"] = np.nan
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pamts = [f"PAY_AMT{i}" for i in range(1, 7)]
    df["pay_max"] = df[pays].max(axis=1)
    df["pay_mean"] = df[pays].mean(axis=1)
    df["pay_sum"] = df[pays].sum(axis=1)
    df["pay_delayed_cnt"] = (df[pays] > 0).sum(axis=1)
    df["bill_sum"] = df[bills].sum(axis=1)
    df["bill_mean"] = df[bills].mean(axis=1)
    df["pamt_sum"] = df[pamts].sum(axis=1)
    df["pamt_mean"] = df[pamts].mean(axis=1)
    df["pay_ratio"] = df["pamt_sum"] / (df["bill_sum"].abs() + 1)
    df["util1"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    df["util_mean"] = df["bill_mean"] / df["LIMIT_BAL"]
    for i in range(1, 7):
        df[f"ratio{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"].abs() + 1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["unpaid1"] = df["BILL_AMT1"] - df["PAY_AMT1"]
    df["unpaid_lim"] = df["unpaid1"] / df["LIMIT_BAL"]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


y = train[TARGET].values
X = fe(train.drop(columns=[TARGET]))
Xt = fe(test[train.drop(columns=[TARGET]).columns])
med = X.median()
Xf = X.fillna(med)
Xtf = Xt.fillna(med)


def make_models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.6, reg_lambda=5, random_state=SEED,
                                  n_jobs=-1, verbose=-1),
        "cat": CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                                  random_seed=SEED, verbose=0, thread_count=-1),
        "xgb": xgb.XGBClassifier(n_estimators=400, learning_rate=0.02, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                                 reg_lambda=5, random_state=SEED, n_jobs=-1,
                                 eval_metric="logloss"),
    }


oof = np.zeros((len(Xf), 3))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr, va in skf.split(Xf, y):
    ms = make_models()
    for j, (name, m) in enumerate(ms.items()):
        m.fit(Xf.iloc[tr], y[tr])
        oof[va, j] = m.predict_proba(Xf.iloc[va])[:, 1]

oof_mean = oof.mean(axis=1)
best_t, best_f = 0.35, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof_mean >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

preds = []
for name, m in make_models().items():
    m.fit(Xf, y)
    preds.append(m.predict_proba(Xtf)[:, 1])
proba = np.mean(preds, axis=0)
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
