import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].astype(int).values
X = train.drop(columns=["y"])
Xt = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]


def fe(df):
    df = df.copy()
    df["pdays_none"] = (df["pdays"] == 999).astype(float)
    df.loc[df["pdays"].isna(), "pdays_none"] = np.nan
    df["emp_euri"] = df["nr.employed"] * df["euribor3m"]
    df["age_missing"] = df["age"].isna().astype(int)
    return df


X = fe(X)
Xt = fe(Xt)
num_cols = [c for c in X.columns if c not in cat_cols]

# LightGBM version: category dtype
all_df = pd.concat([X, Xt], axis=0, keys=["tr", "te"])
X_l = X.copy()
Xt_l = Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(all_df[c]).categories
    X_l[c] = pd.Categorical(X[c], categories=cats)
    Xt_l[c] = pd.Categorical(Xt[c], categories=cats)

# CatBoost version
X_c = X.copy()
Xt_c = Xt.copy()
for c in cat_cols:
    X_c[c] = X_c[c].astype(str)
    Xt_c[c] = Xt_c[c].astype(str)

lgb_params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5, random_state=SEED,
                  verbose=-1, n_jobs=-1)
cb_params = dict(iterations=700, learning_rate=0.04, depth=6,
                 random_seed=SEED, verbose=0, thread_count=-1,
                 cat_features=cat_cols)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
pt = np.zeros(len(Xt))
for tr, va in skf.split(X, y):
    m1 = lgb.LGBMClassifier(**lgb_params)
    m1.fit(X_l.iloc[tr], y[tr])
    p1 = m1.predict_proba(X_l.iloc[va])[:, 1]
    t1 = m1.predict_proba(Xt_l)[:, 1]

    m2 = CatBoostClassifier(**cb_params)
    m2.fit(X_c.iloc[tr], y[tr])
    p2 = m2.predict_proba(X_c.iloc[va])[:, 1]
    t2 = m2.predict_proba(Xt_c)[:, 1]

    oof[va] = (p1 + p2) / 2
    pt += (t1 + t2) / 2 / skf.n_splits

# threshold optimization on OOF
best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.7, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(pt, 0, 1),
                    "label": (pt >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
