import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
X = train.drop(columns=["Revenue"])
Xt = test.copy()

month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def prep(df):
    df = df.copy()
    df["Month_num"] = df["Month"].map(month_map).fillna(0)
    df["Weekend"] = df["Weekend"].astype(int)
    df["VisitorType"] = df["VisitorType"].astype(str)
    df["Month"] = df["Month"].astype(str)
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDur"] = df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    df["DurPerPage"] = df["TotalDur"] / (df["TotalPages"] + 1)
    df["ProdDurPerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["LogPV"] = np.log1p(df["PageValues"])
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["ExitBounceDiff"] = df["ExitRates"] - df["BounceRates"]
    return df


X = prep(X)
Xt = prep(Xt)

# one-hot version for LGBM / LR
full = pd.concat([X, Xt], keys=["tr", "te"])
oh = pd.get_dummies(full, columns=["Month", "VisitorType"], dtype=float)
Xo, Xto = oh.loc["tr"].reset_index(drop=True), oh.loc["te"].reset_index(drop=True)

cat_cols = ["Month", "VisitorType"]
Xc, Xtc = X.reset_index(drop=True), Xt.reset_index(drop=True)


def make_models():
    return {
        "lgb": LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                              random_state=SEED, verbose=-1, n_jobs=-1),
        "cat": CatBoostClassifier(iterations=500, learning_rate=0.04, depth=6, random_seed=SEED,
                                  verbose=0, cat_features=cat_cols, thread_count=-1),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)),
    }


def data_for(name, tr_idx=None, which="train"):
    if name == "cat":
        return (Xc, Xtc)
    return (Xo, Xto)


weights = {"lgb": 0.4, "cat": 0.5, "lr": 0.1}
oof = np.zeros(len(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr, va in skf.split(Xo, y):
    p = np.zeros(len(va))
    for name, m in make_models().items():
        A, _ = data_for(name)
        m.fit(A.iloc[tr], y[tr])
        p += weights[name] * m.predict_proba(A.iloc[va])[:, 1]
    oof[va] = p

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1", best_f, "thr", best_t)

# final fit on all data
proba = np.zeros(len(Xto))
for name, m in make_models().items():
    A, B = data_for(name)
    m.fit(A, y)
    proba += weights[name] * m.predict_proba(B)[:, 1]

proba = np.clip(proba, 0, 1)
out = pd.DataFrame({"proba": proba, "label": (proba >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
