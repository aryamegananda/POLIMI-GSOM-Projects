import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].values
Xtr = train.drop(columns=["Revenue"])
Xte = test.copy()

neg_cols = ["Administrative", "Administrative_Duration", "Informational",
            "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay"]

def fe(df):
    df = df.copy()
    for c in neg_cols:
        df[c + "_neg"] = (df[c] < 0).astype(int)
        df[c] = df[c].abs()
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDur"] = df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    df["DurPerPage"] = df["TotalDur"] / (df["TotalPages"] + 1)
    df["ProdDurPerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["LogPV"] = np.log1p(df["PageValues"])
    df["LogProdDur"] = np.log1p(df["ProductRelated_Duration"])
    df["ExitBounceDiff"] = df["ExitRates"] - df["BounceRates"]
    df["PV_x_Exit"] = df["PageValues"] * (1 - df["ExitRates"])
    return df

Xtr = fe(Xtr)
Xte = fe(Xte)
all_ = pd.concat([Xtr, Xte], keys=["tr", "te"])
all_ = pd.get_dummies(all_, columns=["Month", "VisitorType"], dtype=int)
Xtr = all_.loc["tr"].reset_index(drop=True)
Xte = all_.loc["te"].reset_index(drop=True)
Xte = Xte[Xtr.columns]

def make_models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=2.0,
                                  random_state=SEED, verbose=-1, n_jobs=-1),
        "xgb": xgb.XGBClassifier(n_estimators=400, learning_rate=0.02, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                                 reg_lambda=2.0, random_state=SEED, n_jobs=-1,
                                 eval_metric="logloss"),
        "cat": CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                                  random_seed=SEED, verbose=0, thread_count=-1),
    }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {k: np.zeros(len(Xtr)) for k in make_models()}
for tr_i, va_i in skf.split(Xtr, y):
    models = make_models()
    for k, m in models.items():
        m.fit(Xtr.iloc[tr_i], y[tr_i])
        oof[k][va_i] = m.predict_proba(Xtr.iloc[va_i])[:, 1]

oof_mean = np.mean([oof[k] for k in oof], axis=0)
best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.8, 0.01):
    f = f1_score(y, (oof_mean >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t

models = make_models()
preds = []
for k, m in models.items():
    m.fit(Xtr, y)
    preds.append(m.predict_proba(Xte)[:, 1])
proba = np.mean(preds, axis=0)
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
