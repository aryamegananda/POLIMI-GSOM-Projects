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

y = train["y"].values
X = train.drop(columns=["y"])
Xt = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]


def fe(df):
    df = df.copy()
    df["pdays_never"] = (df["pdays"] == 999).astype(float)
    df.loc[df["pdays"].isna(), "pdays_never"] = np.nan
    df["pdays_c"] = df["pdays"].replace(999, np.nan)
    df["emp_eur"] = df["nr.employed"] / (df["euribor3m"] + 0.1)
    return df


X = fe(X)
Xt = fe(Xt)
num_cols = [c for c in X.columns if c not in cat_cols]

# LightGBM frame (categories)
all_df = pd.concat([X, Xt], axis=0)
X_l, Xt_l = X.copy(), Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(all_df[c]).categories
    X_l[c] = pd.Categorical(X[c], categories=cats)
    Xt_l[c] = pd.Categorical(Xt[c], categories=cats)

# CatBoost frame
X_c, Xt_c = X.copy(), Xt.copy()

lgb_params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED,
                  verbose=-1, n_jobs=-1, cat_smooth=20)
cb_params = dict(iterations=600, learning_rate=0.05, depth=6,
                 random_seed=SEED, verbose=0, thread_count=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(len(X))
oof_c = np.zeros(len(X))
pt_l = np.zeros(len(Xt))
pt_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_l.iloc[tr], y[tr])
    oof_l[va] = m.predict_proba(X_l.iloc[va])[:, 1]
    pt_l += m.predict_proba(Xt_l)[:, 1] / 5

    c = CatBoostClassifier(**cb_params, cat_features=cat_cols)
    Xtr = X_c.iloc[tr].copy()
    c.fit(Xtr, y[tr])
    oof_c[va] = c.predict_proba(X_c.iloc[va])[:, 1]
    pt_c += c.predict_proba(Xt_c)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
pt = 0.5 * pt_l + 0.5 * pt_c

best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(pt, 0, 1),
                    "label": (pt >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
