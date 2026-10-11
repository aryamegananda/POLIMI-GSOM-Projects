import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

TARGET = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

def fe(df):
    d = df.drop(columns=["ID", TARGET], errors="ignore").copy()
    pay = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill = [f"BILL_AMT{i}" for i in range(1, 7)]
    amt = [f"PAY_AMT{i}" for i in range(1, 7)]
    d["EDUCATION"] = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    d["MARRIAGE"] = d["MARRIAGE"].replace({0: 3})
    d["pay_mean"] = d[pay].mean(axis=1)
    d["pay_max"] = d[pay].max(axis=1)
    d["pay_sum_pos"] = d[pay].clip(lower=0).sum(axis=1)
    d["n_delay"] = (d[pay] > 0).sum(axis=1)
    d["bill_mean"] = d[bill].mean(axis=1)
    d["amt_mean"] = d[amt].mean(axis=1)
    d["amt_sum"] = d[amt].sum(axis=1)
    d["bill_sum"] = d[bill].sum(axis=1)
    d["pay_ratio"] = d["amt_sum"] / (d["bill_sum"].abs() + 1)
    for i in range(1, 7):
        d[f"util{i}"] = d[f"BILL_AMT{i}"] / d["LIMIT_BAL"]
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i}"].abs() + 1)
    d["util_mean"] = d["bill_mean"] / d["LIMIT_BAL"]
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["pay_trend"] = d["PAY_0"] - d["PAY_6"]
    return d

X = fe(train)
y = train[TARGET].values
Xt = fe(test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = {k: np.zeros(len(X)) for k in ["lgb", "cat", "lr"]}
tp = {k: np.zeros(len(Xt)) for k in oof}

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=5, random_state=42,
                           verbose=-1, n_jobs=-1)
    m.fit(X.iloc[tr], y[tr])
    oof["lgb"][va] = m.predict_proba(X.iloc[va])[:, 1]
    tp["lgb"] += m.predict_proba(Xt)[:, 1] / 5

    c = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                           random_seed=42, verbose=0, thread_count=-1)
    c.fit(X.iloc[tr], y[tr])
    oof["cat"][va] = c.predict_proba(X.iloc[va])[:, 1]
    tp["cat"] += c.predict_proba(Xt)[:, 1] / 5

    Xs = X.copy().replace([np.inf, -np.inf], 0)
    Xts = Xt.copy().replace([np.inf, -np.inf], 0)
    l = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=42))
    # clip extreme values for linear model stability
    l.fit(Xs.iloc[tr].clip(-1e6, 1e6), y[tr])
    oof["lr"][va] = l.predict_proba(Xs.iloc[va].clip(-1e6, 1e6))[:, 1]
    tp["lr"] += l.predict_proba(Xts.clip(-1e6, 1e6))[:, 1] / 5

w = {"lgb": 0.4, "cat": 0.5, "lr": 0.1}
oof_b = sum(w[k] * oof[k] for k in w)
test_b = sum(w[k] * tp[k] for k in w)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof_b >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(test_b, 0, 1),
                    "label": (test_b >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
