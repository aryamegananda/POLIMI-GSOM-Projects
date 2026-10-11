import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
Xtr = train.drop(columns=["Revenue"]).copy()
Xte = test.copy()
if "Revenue" in Xte.columns:
    Xte = Xte.drop(columns=["Revenue"])

month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def fe(df):
    df = df.copy()
    df["MonthNum"] = df["Month"].map(month_map).fillna(0)
    df["VisitorNew"] = (df["VisitorType"] == "New_Visitor").astype(int)
    df["VisitorRet"] = (df["VisitorType"] == "Returning_Visitor").astype(int)
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDur"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                      + df["ProductRelated_Duration"])
    df["DurPerPage"] = df["TotalDur"] / (df["TotalPages"] + 1)
    df["PV_log"] = np.log1p(df["PageValues"])
    df["HasPV"] = (df["PageValues"] > 0).astype(int)
    df["ExitBounce"] = df["ExitRates"] - df["BounceRates"]
    df["PRdurPerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    return df


Xtr = fe(Xtr)
Xte = fe(Xte)

# numeric version for LGBM / LR
cat_cols = ["Month", "VisitorType"]
Xtr_num = pd.get_dummies(Xtr, columns=cat_cols).astype(float)
Xte_num = pd.get_dummies(Xte, columns=cat_cols).astype(float)
Xte_num = Xte_num.reindex(columns=Xtr_num.columns, fill_value=0)

# catboost version
Xtr_cb = Xtr.copy()
Xte_cb = Xte.copy()


def make_models():
    return {
        "lgb": LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=2.0,
                              random_state=SEED, verbose=-1, n_jobs=-1),
        "cb": CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                                 random_seed=SEED, verbose=0, cat_features=cat_cols,
                                 thread_count=-1),
    }


def fit_predict(name, m, tr_idx, Xa_num, Xa_cb, ya, Xb_num, Xb_cb):
    if name == "cb":
        m.fit(Xa_cb, ya)
        return m.predict_proba(Xb_cb)[:, 1]
    m.fit(Xa_num, ya)
    return m.predict_proba(Xb_num)[:, 1]


# OOF for threshold selection
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
for tr_i, va_i in skf.split(Xtr_num, y):
    preds = []
    for name, m in make_models().items():
        p = fit_predict(name, m, tr_i,
                        Xtr_num.iloc[tr_i], Xtr_cb.iloc[tr_i], y[tr_i],
                        Xtr_num.iloc[va_i], Xtr_cb.iloc[va_i])
        preds.append(p)
    oof[va_i] = np.mean(preds, axis=0)

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.75, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF best F1", best_f, "threshold", best_t)

# final fit on full train
preds = []
for name, m in make_models().items():
    p = fit_predict(name, m, None, Xtr_num, Xtr_cb, y, Xte_num, Xte_cb)
    preds.append(p)
proba = np.mean(preds, axis=0)
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
