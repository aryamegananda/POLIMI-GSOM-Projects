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
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pamt = [f"PAY_AMT{i}" for i in range(1, 7)]
    d["EDUCATION"] = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    d["MARRIAGE"] = d["MARRIAGE"].replace({0: 3})
    d["pay_mean"] = d[pays].mean(axis=1)
    d["pay_max"] = d[pays].max(axis=1)
    d["pay_min"] = d[pays].min(axis=1)
    d["pay_late_cnt"] = (d[pays] > 0).sum(axis=1)
    d["pay_sum_pos"] = d[pays].clip(lower=0).sum(axis=1)
    d["bill_mean"] = d[bills].mean(axis=1)
    d["bill_max"] = d[bills].max(axis=1)
    d["pamt_mean"] = d[pamt].mean(axis=1)
    d["pamt_sum"] = d[pamt].sum(axis=1)
    d["bill_sum"] = d[bills].sum(axis=1)
    d["pay_ratio"] = d["pamt_sum"] / (d["bill_sum"].abs() + 1)
    d["util1"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    d["util_mean"] = d["bill_mean"] / d["LIMIT_BAL"]
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    for i in range(1, 6):
        d[f"ratio{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i+1}"].abs() + 1)
    d["pay_trend"] = d["PAY_0"] - d["PAY_6"]
    return d

X = fe(train)
y = train[TARGET].values
Xt = fe(test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {"lgb": np.zeros(len(X)), "cat": np.zeros(len(X)), "lr": np.zeros(len(X))}
tp = {k: np.zeros(len(Xt)) for k in models}

def make(name):
    if name == "lgb":
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.6, reg_lambda=5, random_state=42,
                                  verbose=-1, n_jobs=-1)
    if name == "cat":
        return CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6,
                                  random_seed=42, verbose=0, thread_count=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=42))

def prep_lr(df):
    d = df.copy()
    for c in d.columns:
        if d[c].abs().max() > 1000:
            d[c] = np.sign(d[c]) * np.log1p(d[c].abs())
    return d

Xl, Xtl = prep_lr(X), prep_lr(Xt)
for tr, va in skf.split(X, y):
    for name in models:
        m = make(name)
        A, B = (Xl, Xtl) if name == "lr" else (X, Xt)
        m.fit(A.iloc[tr], y[tr])
        models[name][va] = m.predict_proba(A.iloc[va])[:, 1]
        tp[name] += m.predict_proba(B)[:, 1] / 5

w = {"lgb": 0.4, "cat": 0.45, "lr": 0.15}
oof = sum(w[k] * models[k] for k in w)
tproba = sum(w[k] * tp[k] for k in w)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof > t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

pd.DataFrame({"proba": tproba, "label": (tproba > best_t).astype(int)}).to_csv("predictions.csv", index=False)
