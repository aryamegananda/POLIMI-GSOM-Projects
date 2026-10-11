import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

def prep(df):
    df = df.copy()
    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
                 "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df["MonthNum"] = df["Month"].map(month_map).fillna(0)
    df["VisitorType"] = df["VisitorType"].astype(str)
    df["Weekend"] = df["Weekend"].astype(int)
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    df["DurPerPage"] = df["TotalDuration"] / (df["TotalPages"] + 1)
    df["PRDurPerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["PV_log"] = np.log1p(df["PageValues"])
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["ExitBounce"] = df["ExitRates"] - df["BounceRates"]
    df["PV_x_Exit"] = df["PageValues"] * (1 - df["ExitRates"])
    return df

tr = prep(train)
te = prep(test)
y = tr["Revenue"].values
X = tr.drop(columns=["Revenue"])
Xt = te[X.columns]

cat_cols = ["Month", "VisitorType"]

# LightGBM version: category dtype
def lgb_frame(df):
    d = df.copy()
    for c in cat_cols:
        d[c] = pd.Categorical(d[c], categories=sorted(train[c].astype(str).unique()))
    return d

Xl, Xtl = lgb_frame(X), lgb_frame(Xt)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
pred = np.zeros(len(Xt))

for tri, vai in skf.split(X, y):
    # LightGBM
    m1 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                            min_child_samples=30, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.7, reg_lambda=5, random_state=42,
                            verbose=-1)
    m1.fit(Xl.iloc[tri], y[tri])
    p1v = m1.predict_proba(Xl.iloc[vai])[:, 1]
    p1t = m1.predict_proba(Xtl)[:, 1]

    # CatBoost
    m2 = CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6,
                            random_seed=42, verbose=0, cat_features=cat_cols,
                            thread_count=4)
    m2.fit(X.iloc[tri], y[tri])
    p2v = m2.predict_proba(X.iloc[vai])[:, 1]
    p2t = m2.predict_proba(Xt)[:, 1]

    oof[vai] = (p1v + p2v) / 2
    pred += (p1t + p2t) / 2 / skf.get_n_splits()

# threshold optimization
best_t, best_f = 0.5, 0
for t in np.arange(0.2, 0.7, 0.01):
    f = f1_score(y, (oof > t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(pred, 0, 1),
                    "label": (pred > best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
