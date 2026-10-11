import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
X = train.drop(columns=["Revenue"])
Xt = test.copy()[X.columns]

num_cols = [c for c in X.columns if c not in ["Month", "VisitorType"]]
month_map = {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8,
             "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def prep(df):
    df = df.copy()
    for c in num_cols:
        df[c + "_isna"] = df[c].isna().astype(int) if c in (
            "PageValues", "ExitRates", "BounceRates") else 0
    df = df.drop(columns=[c for c in df.columns if c.endswith("_isna") and df[c].sum() == 0])
    df["Month_num"] = df["Month"].map(month_map)
    df["Month_num"] = df["Month_num"].fillna(0)
    df["VisitorType"] = df["VisitorType"].fillna("Unknown")
    df["Month"] = df["Month"].fillna("Unknown")
    df["Admin_total"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["Dur_total"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                       + df["ProductRelated_Duration"])
    df["PR_per_page"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["PV_x_exit"] = df["PageValues"] * (1 - df["ExitRates"])
    df["PV_log"] = np.log1p(df["PageValues"])
    return df


Xa = prep(X)
Xb = prep(Xt)
Xb = Xb.reindex(columns=Xa.columns, fill_value=0)

# One-hot version (for LGB and LR), native cat version for CatBoost
full = pd.concat([Xa, Xb], keys=["a", "b"])
oh = pd.get_dummies(full, columns=["Month", "VisitorType"], dtype=float)
Xa_oh, Xb_oh = oh.loc["a"].reset_index(drop=True), oh.loc["b"].reset_index(drop=True)

cat_cols = ["Month", "VisitorType"]
Xa_cb, Xb_cb = Xa.copy(), Xb.copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {k: np.zeros(len(y)) for k in ["lgb", "cb", "lr"]}
pt = {k: np.zeros(len(Xb)) for k in ["lgb", "cb", "lr"]}


def make_lgb():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=5, random_state=SEED,
                              verbose=-1, n_jobs=4)


def make_cb():
    return CatBoostClassifier(iterations=600, learning_rate=0.04, depth=5,
                              random_seed=SEED, verbose=0, thread_count=4,
                              cat_features=cat_cols)


def make_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000, random_state=SEED))


for tr, va in skf.split(Xa_oh, y):
    m = make_lgb().fit(Xa_oh.iloc[tr], y[tr])
    oof["lgb"][va] = m.predict_proba(Xa_oh.iloc[va])[:, 1]
    pt["lgb"] += m.predict_proba(Xb_oh)[:, 1] / 5

    m = make_cb().fit(Xa_cb.iloc[tr], y[tr])
    oof["cb"][va] = m.predict_proba(Xa_cb.iloc[va])[:, 1]
    pt["cb"] += m.predict_proba(Xb_cb)[:, 1] / 5

    m = make_lr().fit(Xa_oh.iloc[tr], y[tr])
    oof["lr"][va] = m.predict_proba(Xa_oh.iloc[va])[:, 1]
    pt["lr"] += m.predict_proba(Xb_oh)[:, 1] / 5

w = {"lgb": 0.4, "cb": 0.5, "lr": 0.1}
oof_b = sum(w[k] * oof[k] for k in w)
test_b = sum(w[k] * pt[k] for k in w)

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof_b >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(test_b, 0, 1),
                    "label": (test_b >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
