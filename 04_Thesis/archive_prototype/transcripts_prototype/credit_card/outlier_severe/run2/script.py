import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

TARGET = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

def fe(df):
    d = df.drop(columns=[c for c in [TARGET, "ID"] if c in df.columns]).copy()
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pamt = [f"PAY_AMT{i}" for i in range(1, 7)]
    d["pay_mean"] = d[pays].mean(axis=1)
    d["pay_max"] = d[pays].max(axis=1)
    d["pay_min"] = d[pays].min(axis=1)
    d["pay_sum"] = d[pays].sum(axis=1)
    d["n_late"] = (d[pays] > 0).sum(axis=1)
    d["n_late2"] = (d[pays] >= 2).sum(axis=1)
    d["bill_mean"] = d[bills].mean(axis=1)
    d["pamt_mean"] = d[pamt].mean(axis=1)
    d["pamt_sum"] = d[pamt].sum(axis=1)
    lim = d["LIMIT_BAL"].abs() + 1
    d["util1"] = d["BILL_AMT1"] / lim
    d["util_mean"] = d["bill_mean"] / lim
    for i in range(1, 7):
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i}"].abs() + 1)
    d["pay_bill_ratio"] = d["pamt_sum"] / (d[bills].sum(axis=1).abs() + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["weird"] = ((d["PAY_0"] == 9) | (d["LIMIT_BAL"] < 0) | (d["AGE"] < 18)).astype(int)
    d["neg_limit"] = (d["LIMIT_BAL"] < 0).astype(int)
    return d

X = fe(train)
y = train[TARGET].values
Xt = fe(test)[X.columns]

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros((len(X), 2))
tp = np.zeros((len(Xt), 2))

for tr, va in skf.split(X, y):
    m1 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                            min_child_samples=40, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.6, reg_lambda=5, random_state=42,
                            verbose=-1, n_jobs=-1)
    m1.fit(X.iloc[tr], y[tr])
    oof[va, 0] = m1.predict_proba(X.iloc[va])[:, 1]
    tp[:, 0] += m1.predict_proba(Xt)[:, 1] / 5

    m2 = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                            random_seed=42, verbose=0, thread_count=-1)
    m2.fit(X.iloc[tr], y[tr])
    oof[va, 1] = m2.predict_proba(X.iloc[va])[:, 1]
    tp[:, 1] += m2.predict_proba(Xt)[:, 1] / 5

oof_p = oof.mean(axis=1)
test_p = tp.mean(axis=1)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof_p >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1", best_f, "threshold", best_t)

out = pd.DataFrame({"proba": np.clip(test_p, 0, 1),
                    "label": (test_p >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
