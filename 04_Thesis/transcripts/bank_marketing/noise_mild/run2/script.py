import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].values
X = train.drop(columns=["y"]).copy()
Xt = test.copy()[X.columns]

cat_cols = [c for c in X.columns if X[c].dtype == object]


def fe(df):
    df = df.copy()
    df["prev_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, np.nan)
    df["euribor_x_emp"] = df["euribor3m"] * df["emp.var.rate"]
    df["nr_per_euribor"] = df["nr.employed"] / (df["euribor3m"] + 1)
    df["age_campaign"] = df["age"] / (df["campaign"] + 1)
    return df


X = fe(X)
Xt = fe(Xt)

# LightGBM version: categorical dtype
X_l, Xt_l = X.copy(), Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], Xt[c]])).categories
    X_l[c] = pd.Categorical(X[c], categories=cats)
    Xt_l[c] = pd.Categorical(Xt[c], categories=cats)

# CatBoost version: strings, NaN filled
X_c, Xt_c = X.copy(), Xt.copy()
X_c["pdays"] = X_c["pdays"].fillna(-1)
Xt_c["pdays"] = Xt_c["pdays"].fillna(-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X))
oof_c = np.zeros(len(X))
pred_l = np.zeros(len(Xt))
pred_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.02, num_leaves=15,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5.0, cat_smooth=20,
        random_state=42, verbose=-1, n_jobs=-1)
    m.fit(X_l.iloc[tr], y[tr])
    oof_l[va] = m.predict_proba(X_l.iloc[va])[:, 1]
    pred_l += m.predict_proba(Xt_l)[:, 1] / 5

    c = CatBoostClassifier(
        iterations=600, learning_rate=0.04, depth=6, l2_leaf_reg=5,
        random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1)
    c.fit(X_c.iloc[tr], y[tr])
    oof_c[va] = c.predict_proba(X_c.iloc[va])[:, 1]
    pred_c += c.predict_proba(Xt_c)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
proba = 0.5 * pred_l + 0.5 * pred_c

best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.7, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "OOF F1", best_f)

out = pd.DataFrame({
    "proba": np.clip(proba, 0, 1),
    "label": (proba >= best_t).astype(int),
})
out.to_csv("predictions.csv", index=False)
