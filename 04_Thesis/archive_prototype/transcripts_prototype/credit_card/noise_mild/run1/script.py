import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from catboost import CatBoostClassifier

target = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

def fe(df):
    d = df.drop(columns=["ID", target], errors="ignore").copy()
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pa = [f"PAY_AMT{i}" for i in range(1, 7)]
    d["pay_mean"] = d[pays].mean(axis=1)
    d["pay_max"] = d[pays].max(axis=1)
    d["pay_pos_cnt"] = (d[pays] > 0).sum(axis=1)
    d["pay_sum_pos"] = d[pays].clip(lower=0).sum(axis=1)
    d["bill_mean"] = d[bills].mean(axis=1)
    d["payamt_mean"] = d[pa].mean(axis=1)
    d["payamt_sum"] = d[pa].sum(axis=1)
    d["util1"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    d["util_mean"] = d["bill_mean"] / d["LIMIT_BAL"]
    d["pay_ratio"] = d["payamt_sum"] / (d[bills].sum(axis=1).abs() + 1)
    for i in range(1, 6):
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i+1}"].abs() + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["limit_per_age"] = d["LIMIT_BAL"] / d["AGE"]
    return d

X = fe(train)
y = train[target].values
Xt = fe(test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
names = ["lgb", "cat", "lr"]
oof = {n: np.zeros(len(X)) for n in names}
tp = {n: np.zeros(len(Xt)) for n in names}

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5, random_state=42,
                           verbose=-1, n_jobs=-1)
    m.fit(X.iloc[tr], y[tr])
    oof["lgb"][va] = m.predict_proba(X.iloc[va])[:, 1]
    tp["lgb"] += m.predict_proba(Xt)[:, 1] / 5

    c = CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6,
                           random_seed=42, verbose=0, thread_count=-1)
    c.fit(X.iloc[tr], y[tr])
    oof["cat"][va] = c.predict_proba(X.iloc[va])[:, 1]
    tp["cat"] += c.predict_proba(Xt)[:, 1] / 5

    Xs = X.replace([np.inf, -np.inf], 0)
    Xts = Xt.replace([np.inf, -np.inf], 0)
    l = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=42))
    l.fit(Xs.iloc[tr], y[tr])
    oof["lr"][va] = l.predict_proba(Xs.iloc[va])[:, 1]
    tp["lr"] += l.predict_proba(Xts)[:, 1] / 5

w = {"lgb": 0.4, "cat": 0.4, "lr": 0.2}
oof_b = sum(w[n] * oof[n] for n in names)
test_b = sum(w[n] * tp[n] for n in names)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof_b >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "OOF F1", best_f)

out = pd.DataFrame({"proba": np.clip(test_b, 0, 1),
                    "label": (test_b >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
