import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

target = "Revenue"
y = train[target].astype(int).values
X = train.drop(columns=[target])
Xt = test.copy()[X.columns]

months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def prep(df):
    d = df.copy()
    d["Month"] = d["Month"].astype(str)
    mmap = {m: i for i, m in enumerate(months)}
    d["Month_num"] = d["Month"].map(mmap).fillna(-1)
    d["Visitor_New"] = (d["VisitorType"] == "New_Visitor").astype(int)
    d["Visitor_Ret"] = (d["VisitorType"] == "Returning_Visitor").astype(int)
    d = d.drop(columns=["Month", "VisitorType"])
    d["Weekend"] = d["Weekend"].astype(int)
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDuration"] = d["Administrative_Duration"] + d["Informational_Duration"] + d["ProductRelated_Duration"]
    d["DurPerPage"] = d["TotalDuration"] / (d["TotalPages"] + 1)
    d["PR_DurPerPage"] = d["ProductRelated_Duration"] / (d["ProductRelated"] + 1)
    d["LogPageValues"] = np.log1p(d["PageValues"])
    d["PV_Exit"] = d["PageValues"] * (1 - d["ExitRates"])
    d["Bounce_Exit"] = d["BounceRates"] - d["ExitRates"]
    return d


Xp = prep(X)
Xtp = prep(Xt)
Xtp = Xtp[Xp.columns]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    "lgb": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=2.0,
                              random_state=SEED, verbose=-1, n_jobs=4),
    "hgb": HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250, max_leaf_nodes=15,
                                          min_samples_leaf=30, l2_regularization=1.0,
                                          random_state=SEED),
    "lr": make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)),
}

# Log-transform for LR robustness
oof = {}
for name, m in models.items():
    oof[name] = cross_val_predict(m, Xp, y, cv=skf, method="predict_proba")[:, 1]

weights = {"lgb": 0.45, "hgb": 0.4, "lr": 0.15}
oof_blend = sum(weights[k] * oof[k] for k in models)

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof_blend >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("Best threshold", best_t, "OOF F1", best_f)

test_blend = np.zeros(len(Xtp))
for name, m in models.items():
    m.fit(Xp, y)
    test_blend += weights[name] * m.predict_proba(Xtp)[:, 1]

proba = np.clip(test_blend, 0, 1)
label = (proba >= best_t).astype(int)
pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
