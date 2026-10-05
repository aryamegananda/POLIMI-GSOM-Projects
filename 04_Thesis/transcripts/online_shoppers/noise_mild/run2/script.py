import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

def prep(df):
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6, "Jul": 7,
                 "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df["MonthNum"] = df["Month"].map(month_map).fillna(0)
    df["VisitorType"] = df["VisitorType"].astype(str)
    df = pd.get_dummies(df, columns=["Month", "VisitorType"], dtype=int)
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    df["DurPerPage"] = df["TotalDuration"] / (df["TotalPages"] + 1)
    df["PV_log"] = np.log1p(df["PageValues"])
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["ExitBounce"] = df["ExitRates"] - df["BounceRates"]
    df["PV_x_Exit"] = df["PageValues"] * (1 - df["ExitRates"])
    return df

y = train["Revenue"].values
Xtr = prep(train.drop(columns=["Revenue"]))
Xte = prep(test)
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

def make_models():
    return [
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=30,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
                           random_state=SEED, verbose=-1, n_jobs=4),
        HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250, max_leaf_nodes=15,
                                       min_samples_leaf=30, l2_regularization=2.0, random_state=SEED),
    ]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(Xtr))
n_models = len(make_models())
for tr, va in skf.split(Xtr, y):
    ps = []
    for m in make_models():
        m.fit(Xtr.iloc[tr], y[tr])
        ps.append(m.predict_proba(Xtr.iloc[va])[:, 1])
    oof[va] = np.mean(ps, axis=0)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

ps = []
for m in make_models():
    m.fit(Xtr, y)
    ps.append(m.predict_proba(Xte)[:, 1])
proba = np.mean(ps, axis=0)
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
