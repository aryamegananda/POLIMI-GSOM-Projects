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

months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jul": 7,
          "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def prep(df):
    d = df.copy()
    d["MonthNum"] = d["Month"].map(months).fillna(0)
    d["IsReturning"] = (d["VisitorType"] == "Returning_Visitor").astype(int)
    d["IsNew"] = (d["VisitorType"] == "New_Visitor").astype(int)
    d = d.drop(columns=["Month", "VisitorType"])
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDuration"] = (d["Administrative_Duration"] + d["Informational_Duration"]
                          + d["ProductRelated_Duration"])
    d["AvgProdDur"] = d["ProductRelated_Duration"] / (d["ProductRelated"] + 1)
    d["LogPV"] = np.log1p(d["PageValues"])
    d["HasPV"] = (d["PageValues"] > 0).astype(int)
    d["ExitBounce"] = d["ExitRates"] - d["BounceRates"]
    d["PVxExit"] = d["PageValues"] * (1 - d["ExitRates"])
    for c in ["Administrative_Duration", "Informational_Duration",
              "ProductRelated_Duration", "TotalDuration"]:
        d["log_" + c] = np.log1p(d[c])
    return d


y = train["Revenue"].values
X = prep(train.drop(columns=["Revenue"]))
Xt = prep(test)[X.columns]


def make_models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=2.0,
                                  random_state=SEED, verbose=-1, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250,
                                              max_leaf_nodes=15, min_samples_leaf=30,
                                              l2_regularization=1.0, random_state=SEED),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000,
                                                                 random_state=SEED)),
    }


weights = {"lgb": 0.4, "hgb": 0.4, "lr": 0.2}

# OOF predictions for threshold selection
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
for tr, va in skf.split(X, y):
    p = np.zeros(len(va))
    for name, m in make_models().items():
        m.fit(X.iloc[tr], y[tr])
        p += weights[name] * m.predict_proba(X.iloc[va])[:, 1]
    oof[va] = p

ths = np.arange(0.15, 0.7, 0.01)
f1s = [f1_score(y, (oof > t).astype(int)) for t in ths]
# smooth F1 curve to pick a robust threshold
sm = np.convolve(f1s, np.ones(5) / 5, mode="same")
best_t = ths[2:-2][np.argmax(sm[2:-2])]
print("best threshold", best_t, "oof f1", max(f1s))

# Final fit on all training data
proba = np.zeros(len(Xt))
for name, m in make_models().items():
    m.fit(X, y)
    proba += weights[name] * m.predict_proba(Xt)[:, 1]

out = pd.DataFrame({"proba": np.clip(proba, 0, 1),
                    "label": (proba > best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
