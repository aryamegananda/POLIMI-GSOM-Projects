import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import lightgbm as lgb

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

target = "Revenue"
y = train[target].astype(int).values
Xtr = train.drop(columns=[target]).copy()
Xte = test.copy()
Xte = Xte[Xtr.columns]

month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def prep(df):
    df = df.copy()
    df["MonthNum"] = df["Month"].map(month_map).fillna(0)
    df["VisitorType"] = df["VisitorType"].fillna("Other")
    for v in ["Returning_Visitor", "New_Visitor", "Other"]:
        df["VT_" + v] = (df["VisitorType"] == v).astype(int)
    df = df.drop(columns=["Month", "VisitorType"])
    df["Weekend"] = df["Weekend"].astype(int)
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                           + df["ProductRelated_Duration"])
    df["AvgProdDur"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["ExitBounce"] = df["ExitRates"] - df["BounceRates"]
    df["PV_log"] = np.log1p(df["PageValues"])
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["PV_x_Exit"] = df["PageValues"] * (1 - df["ExitRates"])
    df["LogProdDur"] = np.log1p(df["ProductRelated_Duration"])
    return df


Xtr = prep(Xtr)
Xte = prep(Xte)
Xte = Xte[Xtr.columns]

models = {
    "lgb": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15,
                                      min_child_samples=30, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.7, reg_lambda=5.0,
                                      random_state=SEED, verbose=-1, n_jobs=4),
    "hgb": lambda: HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250,
                                                  max_leaf_nodes=15, min_samples_leaf=30,
                                                  l2_regularization=2.0, random_state=SEED),
    "lr": lambda: make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {}
for name, mk in models.items():
    oof[name] = cross_val_predict(mk(), Xtr, y, cv=skf, method="predict_proba")[:, 1]

weights = {"lgb": 0.4, "hgb": 0.4, "lr": 0.2}
oof_blend = sum(weights[k] * oof[k] for k in models)

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof_blend >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "cv F1", best_f)

test_proba = np.zeros(len(Xte))
for name, mk in models.items():
    m = mk()
    m.fit(Xtr, y)
    test_proba += weights[name] * m.predict_proba(Xte)[:, 1]

test_proba = np.clip(test_proba, 0, 1)
labels = (test_proba >= best_t).astype(int)
pd.DataFrame({"proba": test_proba, "label": labels}).to_csv("predictions.csv", index=False)
