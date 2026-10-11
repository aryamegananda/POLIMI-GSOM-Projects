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
    d = df.drop(columns=[TARGET], errors="ignore").copy()
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    pays = [f"PAY_AMT{i}" for i in range(1, 7)]
    pst = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    d["EDUCATION"] = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    d["MARRIAGE"] = d["MARRIAGE"].replace({0: 3})
    d["bill_sum"] = d[bills].sum(axis=1)
    d["bill_mean"] = d[bills].mean(axis=1)
    d["pay_sum"] = d[pays].sum(axis=1)
    d["pay_mean"] = d[pays].mean(axis=1)
    d["pay_ratio"] = d["pay_sum"] / (d["bill_sum"].abs() + 1)
    d["pst_max"] = d[pst].max(axis=1)
    d["pst_mean"] = d[pst].mean(axis=1)
    d["pst_sum"] = d[pst].sum(axis=1)
    d["n_late"] = (d[pst] > 0).sum(axis=1)
    d["n_revolve"] = (d[pst] >= 0).sum(axis=1)
    d["util1"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    d["util_mean"] = d["bill_mean"] / d["LIMIT_BAL"]
    for i in range(1, 6):
        d[f"pr{i}"] = d[f"PAY_AMT{i}"] / (d[f"BILL_AMT{i+1}"].abs() + 1)
        d[f"bdiff{i}"] = d[f"BILL_AMT{i}"] - d[f"BILL_AMT{i+1}"]
    d["pr_last"] = d["PAY_AMT1"] / (d["BILL_AMT2"].abs() + 1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    d["limit_pay"] = d["LIMIT_BAL"] - d["BILL_AMT1"]
    return d

X = fe(train)
y = train[TARGET].values
Xt = fe(test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = {k: np.zeros(len(X)) for k in ["lgb", "cat", "lr"]}
pt = {k: np.zeros(len(Xt)) for k in oof}

# LR features: signed log transform
def lrprep(d):
    d = d.copy()
    for c in d.columns:
        if c.startswith(("BILL_AMT", "PAY_AMT", "bill_", "pay_sum", "pay_mean", "bdiff", "bill_trend", "limit_pay", "LIMIT_BAL")):
            d[c] = np.sign(d[c]) * np.log1p(np.abs(d[c]))
    d = d.replace([np.inf, -np.inf], 0).fillna(0)
    d["pay_ratio"] = d["pay_ratio"].clip(-10, 10)
    for c in [c for c in d.columns if c.startswith("pr") or c.startswith("util")]:
        d[c] = d[c].clip(-10, 10)
    return d

Xl, Xtl = lrprep(X), lrprep(Xt)

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=5, random_state=42,
                           verbose=-1, n_jobs=-1)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof["lgb"][va] = m.predict_proba(X.iloc[va])[:, 1]
    pt["lgb"] += m.predict_proba(Xt)[:, 1] / 5

    c = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6,
                           random_seed=42, verbose=0, od_type="Iter", od_wait=100,
                           thread_count=-1)
    c.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]), use_best_model=True)
    oof["cat"][va] = c.predict_proba(X.iloc[va])[:, 1]
    pt["cat"] += c.predict_proba(Xt)[:, 1] / 5

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

out = pd.DataFrame({"proba": np.clip(pt_b, 0, 1),
                    "label": (pt_b >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
