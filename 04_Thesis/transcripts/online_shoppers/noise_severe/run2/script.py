import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
Xtr = train.drop(columns=["Revenue"]).copy()
Xte = test.copy()
Xte = Xte[Xtr.columns]

month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def fe(df):
    df = df.copy()
    df["Month_num"] = df["Month"].map(month_map).fillna(0)
    df["Weekend"] = df["Weekend"].astype(int)
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                           + df["ProductRelated_Duration"])
    df["AvgProdDur"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["AvgAdminDur"] = df["Administrative_Duration"] / (df["Administrative"] + 1)
    df["PV_log"] = np.log1p(df["PageValues"])
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["Exit_Bounce"] = df["ExitRates"] - df["BounceRates"]
    df["PV_x_Exit"] = df["PageValues"] * (1 - df["ExitRates"])
    return df


Xtr = fe(Xtr)
Xte = fe(Xte)
cat_cols = ["Month", "VisitorType"]

# numeric/one-hot version for lgb & LR
full = pd.concat([Xtr, Xte], keys=["tr", "te"])
full_oh = pd.get_dummies(full, columns=cat_cols, dtype=float)
for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
    pass
A_tr = full_oh.loc["tr"].reset_index(drop=True)
A_te = full_oh.loc["te"].reset_index(drop=True)

# catboost version
C_tr = Xtr.reset_index(drop=True)
C_te = Xte.reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {k: np.zeros(len(y)) for k in ["lgb", "cat", "lr"]}
pte = {k: np.zeros(len(Xte)) for k in ["lgb", "cat", "lr"]}

lgb_params = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                  random_state=SEED, verbose=-1, n_jobs=-1)
Atr_lr = A_tr.copy()
Ate_lr = A_te.copy()
for c in ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
          "TotalDuration", "AvgProdDur", "AvgAdminDur", "PageValues", "ProductRelated",
          "TotalPages", "Administrative", "Informational"]:
    Atr_lr[c] = np.log1p(Atr_lr[c])
    Ate_lr[c] = np.log1p(Ate_lr[c])

for tr_i, va_i in skf.split(A_tr, y):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(A_tr.iloc[tr_i], y[tr_i])
    oof["lgb"][va_i] = m.predict_proba(A_tr.iloc[va_i])[:, 1]
    pte["lgb"] += m.predict_proba(A_te)[:, 1] / 5

    cb = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, random_seed=SEED,
                            verbose=0, cat_features=cat_cols, thread_count=-1)
    cb.fit(C_tr.iloc[tr_i], y[tr_i])
    oof["cat"][va_i] = cb.predict_proba(C_tr.iloc[va_i])[:, 1]
    pte["cat"] += cb.predict_proba(C_te)[:, 1] / 5

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=SEED))
    lr.fit(Atr_lr.iloc[tr_i], y[tr_i])
    oof["lr"][va_i] = lr.predict_proba(Atr_lr.iloc[va_i])[:, 1]
    pte["lr"] += lr.predict_proba(Ate_lr)[:, 1] / 5

w = {"lgb": 0.4, "cat": 0.5, "lr": 0.1}
oof_ens = sum(w[k] * oof[k] for k in w)
te_ens = sum(w[k] * pte[k] for k in w)

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof_ens >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(te_ens, 0, 1),
                    "label": (te_ens >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
