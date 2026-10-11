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
    df = df.copy()
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pamt = [f"PAY_AMT{i}" for i in range(1, 7)]
    df["pay_mean"] = df[pays].mean(axis=1)
    df["pay_max"] = df[pays].max(axis=1)
    df["pay_sum_pos"] = df[pays].clip(lower=0).sum(axis=1)
    df["n_late"] = (df[pays] > 0).sum(axis=1)
    df["bill_mean"] = df[bills].mean(axis=1)
    df["pamt_mean"] = df[pamt].mean(axis=1)
    df["pamt_sum"] = df[pamt].sum(axis=1)
    df["util1"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    df["util_mean"] = df["bill_mean"] / df["LIMIT_BAL"]
    df["pay_ratio"] = df["pamt_sum"] / (df[bills].clip(lower=0).sum(axis=1) + 1)
    for i in range(1, 6):
        df[f"ratio{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i+1}"].clip(lower=0) + 1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["pay_trend"] = df["PAY_0"] - df["PAY_6"]
    return df

y = train[TARGET].values
X = fe(train.drop(columns=[TARGET]))
Xt = fe(test[train.drop(columns=[TARGET]).columns])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = {k: np.zeros(len(X)) for k in ["lgb", "cat", "lr"]}
pt = {k: np.zeros(len(Xt)) for k in oof}

lgb_params = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
                  random_state=42, verbose=-1, n_jobs=-1)

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**lgb_params).fit(X.iloc[tr], y[tr])
    oof["lgb"][va] = m.predict_proba(X.iloc[va])[:, 1]
    pt["lgb"] += m.predict_proba(Xt)[:, 1] / 5

    c = CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6, random_seed=42,
                           verbose=0, thread_count=-1)
    c.fit(X.iloc[tr], y[tr])
    oof["cat"][va] = c.predict_proba(X.iloc[va])[:, 1]
    pt["cat"] += c.predict_proba(Xt)[:, 1] / 5

    Xl = np.sign(X) * np.log1p(np.abs(X))
    Xtl = np.sign(Xt) * np.log1p(np.abs(Xt))
    l = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=42))
    l.fit(Xl.iloc[tr], y[tr])
    oof["lr"][va] = l.predict_proba(Xl.iloc[va])[:, 1]
    pt["lr"] += l.predict_proba(Xtl)[:, 1] / 5

w = {"lgb": 0.4, "cat": 0.45, "lr": 0.15}
oof_b = sum(w[k] * oof[k] for k in w)
pt_b = sum(w[k] * pt[k] for k in w)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof_b >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

pt_b = np.clip(pt_b, 0, 1)
out = pd.DataFrame({"proba": pt_b, "label": (pt_b >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
