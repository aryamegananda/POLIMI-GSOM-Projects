import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
X = train.drop(columns=["Revenue"])
Xt = test.copy()[X.columns]

months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fe(df):
    df = df.copy()
    df["Month"] = df["Month"].astype(str)
    df["VisitorType"] = df["VisitorType"].astype(str)
    df["Month_num"] = df["Month"].map({m: i for i, m in enumerate(months)}).astype(float)
    df["Visitor_code"] = df["VisitorType"].map(
        {"Returning_Visitor": 0, "New_Visitor": 1, "Other": 2}).astype(float)
    df["Month"] = df["Month"].astype("category")
    df["VisitorType"] = df["VisitorType"].astype("category")
    df["TotalPages"] = df[["Administrative", "Informational", "ProductRelated"]].sum(axis=1, min_count=1)
    df["TotalDur"] = df[["Administrative_Duration", "Informational_Duration",
                         "ProductRelated_Duration"]].sum(axis=1, min_count=1)
    df["DurPerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["PV_log"] = np.log1p(df["PageValues"])
    df["PV_x_exit"] = df["PageValues"] * (1 - df["ExitRates"])
    df["Exit_minus_Bounce"] = df["ExitRates"] - df["BounceRates"]
    df["n_missing"] = df.isna().sum(axis=1)
    return df


allX = pd.concat([X, Xt], axis=0, ignore_index=True)
allF = fe(allX)
Xf = allF.iloc[:len(X)].reset_index(drop=True)
Xtf = allF.iloc[len(X):].reset_index(drop=True)

# numeric-only versions for HGB / catboost
Xn = pd.get_dummies(Xf, columns=["Month", "VisitorType"], dtype=float)
Xtn = pd.get_dummies(Xtf, columns=["Month", "VisitorType"], dtype=float)
Xtn = Xtn.reindex(columns=Xn.columns, fill_value=0)


def make_models():
    return {
        "lgb": lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
            random_state=SEED, verbose=-1, n_jobs=4),
        "hgb": lambda: HistGradientBoostingClassifier(
            learning_rate=0.04, max_iter=300, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, random_state=SEED),
        "cat": lambda: CatBoostClassifier(
            iterations=600, learning_rate=0.04, depth=6, random_seed=SEED,
            verbose=0, thread_count=4),
    }


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {k: np.zeros(len(y)) for k in make_models()}
tp = {k: np.zeros(len(Xtn)) for k in make_models()}

for tr, va in skf.split(Xn, y):
    for name, mk in make_models().items():
        if name == "lgb":
            A, B, C = Xf.iloc[tr], Xf.iloc[va], Xtf
        else:
            A, B, C = Xn.iloc[tr], Xn.iloc[va], Xtn
        m = mk()
        m.fit(A, y[tr])
        oof[name][va] = m.predict_proba(B)[:, 1]
        tp[name] += m.predict_proba(C)[:, 1] / skf.n_splits

oof_ens = np.mean([oof[k] for k in oof], axis=0)
test_ens = np.mean([tp[k] for k in tp], axis=0)

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.75, 0.01):
    f = f1_score(y, (oof_ens >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(test_ens, 0, 1),
                    "label": (test_ens >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
