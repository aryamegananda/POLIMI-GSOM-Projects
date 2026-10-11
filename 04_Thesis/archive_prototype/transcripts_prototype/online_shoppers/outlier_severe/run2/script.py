import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
X = train.drop(columns=["Revenue"])
Xt = test.copy()[X.columns]

months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def prep(df):
    d = df.copy()
    d["Weekend"] = d["Weekend"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["Month_num"] = d["Month"].map({m: i for i, m in enumerate(months)}).fillna(-1)
    d["Month_num"] = d["Month_num"].astype(float)
    for v in ["Returning_Visitor", "New_Visitor", "Other"]:
        d["VT_" + v] = (d["VisitorType"] == v).astype(int)
    for m in d["Month"].unique().tolist() if False else []:
        pass
    d = d.drop(columns=["Month", "VisitorType"])
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDur"] = d["Administrative_Duration"] + d["Informational_Duration"] + d["ProductRelated_Duration"]
    d["DurPerPage"] = d["TotalDur"] / (d["TotalPages"].abs() + 1)
    d["PV_log"] = np.sign(d["PageValues"]) * np.log1p(d["PageValues"].abs())
    d["PV_pos"] = (d["PageValues"] > 0).astype(int)
    d["Exit_Bounce"] = d["ExitRates"] - d["BounceRates"]
    d["PV_x_Exit"] = d["PageValues"] * d["ExitRates"]
    return d

Xp = prep(X)
Xtp = prep(Xt)
Xtp = Xtp[Xp.columns]

def models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=2.0,
                                  random_state=42, verbose=-1, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(learning_rate=0.03, max_iter=250, max_leaf_nodes=15,
                                              min_samples_leaf=30, l2_regularization=1.0,
                                              random_state=42),
        "lr": make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
    }

weights = {"lgb": 0.45, "hgb": 0.4, "lr": 0.15}

# OOF to choose threshold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(Xp))
for tr, va in skf.split(Xp, y):
    p = np.zeros(len(va))
    for name, m in models().items():
        m.fit(Xp.iloc[tr], y[tr])
        p += weights[name] * m.predict_proba(Xp.iloc[va])[:, 1]
    oof[va] = p

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

# Final fit
proba = np.zeros(len(Xtp))
for name, m in models().items():
    m.fit(Xp, y)
    proba += weights[name] * m.predict_proba(Xtp)[:, 1]

proba = np.clip(proba, 0, 1)
label = (proba >= best_t).astype(int)
pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
