import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
months = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def prep(df):
    d = df.drop(columns=["Revenue"], errors="ignore").copy()
    d["Weekend"] = d["Weekend"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    mmap = {m: i for i, m in enumerate(months)}
    d["Month_num"] = d["Month"].map(mmap).fillna(-1)
    d = pd.get_dummies(d, columns=["Month", "VisitorType"], dtype=float)
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDur"] = (d["Administrative_Duration"] + d["Informational_Duration"]
                     + d["ProductRelated_Duration"])
    d["DurPerPage"] = d["ProductRelated_Duration"] / (d["ProductRelated"].abs() + 1)
    d["PV_log"] = np.sign(d["PageValues"]) * np.log1p(d["PageValues"].abs())
    d["PV_pos"] = (d["PageValues"] > 0).astype(int)
    d["ExitBounce"] = d["ExitRates"] - d["BounceRates"]
    return d


Xall = prep(pd.concat([train.drop(columns=["Revenue"]), test], ignore_index=True))
X = Xall.iloc[:len(train)].reset_index(drop=True)
Xt = Xall.iloc[len(train):].reset_index(drop=True)


def make_models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=2.0,
                                  random_state=SEED, verbose=-1, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250, max_leaf_nodes=15,
                                              min_samples_leaf=30, l2_regularization=1.0,
                                              random_state=SEED),
        "lr": make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)),
    }


weights = {"lgb": 0.4, "hgb": 0.4, "lr": 0.2}
oof = np.zeros(len(X))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr, va in skf.split(X, y):
    ms = make_models()
    p = 0
    for k, m in ms.items():
        m.fit(X.iloc[tr], y[tr])
        p = p + weights[k] * m.predict_proba(X.iloc[va])[:, 1]
    oof[va] = p

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("CV F1", best_f, "thr", best_t)

ms = make_models()
pt = 0
for k, m in ms.items():
    m.fit(X, y)
    pt = pt + weights[k] * m.predict_proba(Xt)[:, 1]

pt = np.clip(pt, 0, 1)
out = pd.DataFrame({"proba": pt, "label": (pt >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
