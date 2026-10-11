import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

TARGET = "default.payment.next.month"
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

feats = [c for c in test.columns if c != TARGET]
pay = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
pamt = [f"PAY_AMT{i}" for i in range(1, 7)]


def fe(df):
    d = df[feats].copy()
    d = d.drop(columns=["ID"])  # ID is not informative
    d["pay_mean"] = df[pay].mean(axis=1)
    d["pay_max"] = df[pay].max(axis=1)
    d["pay_min"] = df[pay].min(axis=1)
    d["pay_pos_cnt"] = (df[pay] > 0).sum(axis=1)
    d["pay_sum"] = df[pay].sum(axis=1)
    d["bill_mean"] = df[bill].mean(axis=1)
    d["bill_max"] = df[bill].max(axis=1)
    d["pamt_mean"] = df[pamt].mean(axis=1)
    d["pamt_sum"] = df[pamt].sum(axis=1)
    d["util1"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
    d["util_mean"] = d["bill_mean"] / (df["LIMIT_BAL"] + 1)
    d["pay_ratio1"] = df["PAY_AMT1"] / (df["BILL_AMT1"].abs() + 1)
    d["pay_ratio_all"] = d["pamt_sum"] / (df[bill].abs().sum(axis=1) + 1)
    d["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    for i in range(1, 6):
        d[f"diff{i}"] = df[f"BILL_AMT{i+1}"] - df[f"PAY_AMT{i}"] - df[f"BILL_AMT{i}"]
    d["limit_pamt"] = df["PAY_AMT1"] / (df["LIMIT_BAL"] + 1)
    return d


X = fe(train)
y = train[TARGET].values
Xt = fe(test)

params = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
              random_state=42, verbose=-1, n_jobs=-1)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
pt = np.zeros(len(Xt))
for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[tr], y[tr])
    p_va = m.predict_proba(X.iloc[va])[:, 1]
    p_te = m.predict_proba(Xt)[:, 1]

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.5, max_iter=2000, random_state=42))
    Xtr_l = X.iloc[tr].replace([np.inf, -np.inf], np.nan)
    lr.fit(Xtr_l, y[tr])
    l_va = lr.predict_proba(X.iloc[va].replace([np.inf, -np.inf], np.nan))[:, 1]
    l_te = lr.predict_proba(Xt.replace([np.inf, -np.inf], np.nan))[:, 1]

    oof[va] = 0.8 * p_va + 0.2 * l_va
    pt += (0.8 * p_te + 0.2 * l_te) / 5

best_t, best_f = 0.35, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof > t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("CV F1", best_f, "threshold", best_t)

out = pd.DataFrame({"proba": np.clip(pt, 0, 1), "label": (pt > best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
