import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def prep(df):
    d = df.copy()
    d["Month"] = d["Month"].map(month_map).fillna(0).astype(int)
    d["Weekend"] = d["Weekend"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["VT_New"] = (d["VisitorType"] == "New_Visitor").astype(int)
    d["VT_Ret"] = (d["VisitorType"] == "Returning_Visitor").astype(int)
    d = d.drop(columns=["VisitorType"])
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDur"] = d["Administrative_Duration"] + d["Informational_Duration"] + d["ProductRelated_Duration"]
    d["DurPerPage"] = d["TotalDur"] / (d["TotalPages"] + 1)
    d["PV_log"] = np.log1p(d["PageValues"])
    d["PV_x_exit"] = d["PageValues"] * (1 - d["ExitRates"])
    d["Prod_dur_per"] = d["ProductRelated_Duration"] / (d["ProductRelated"] + 1)
    d["Bounce_Exit"] = d["BounceRates"] - d["ExitRates"]
    return d


y = train["Revenue"].values
X = prep(train.drop(columns=["Revenue"]))
Xt = prep(test)[X.columns]


def make_models():
    return [
        LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                       random_state=42, verbose=-1, n_jobs=-1),
        CatBoostClassifier(iterations=500, learning_rate=0.04, depth=6, random_seed=42,
                           verbose=0, thread_count=-1),
        HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250, max_leaf_nodes=15,
                                       min_samples_leaf=30, l2_regularization=1.0,
                                       random_state=42),
    ]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
for tr, va in skf.split(X, y):
    ps = []
    for m in make_models():
        m.fit(X.iloc[tr], y[tr])
        ps.append(m.predict_proba(X.iloc[va])[:, 1])
    oof[va] = np.mean(ps, axis=0)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("CV best threshold", best_t, "F1", best_f)

ps = []
for m in make_models():
    m.fit(X, y)
    ps.append(m.predict_proba(Xt)[:, 1])
proba = np.mean(ps, axis=0)
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
