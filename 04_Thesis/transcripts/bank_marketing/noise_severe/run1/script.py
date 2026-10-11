import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].values
X = train.drop(columns=["y"])
Xt = test[X.columns].copy()

cat_cols = [c for c in X.columns if X[c].dtype == object]


def add_feats(df):
    df = df.copy()
    df["pdays_flag"] = (df["pdays"] == 999).astype(int)
    df["pdays_clean"] = df["pdays"].where(df["pdays"] != 999, -1)
    df["n_unknown"] = (df[cat_cols] == "unknown").sum(axis=1)
    df["euribor_emp"] = df["euribor3m"] * df["nr.employed"] / 5000
    df["age_bin"] = pd.cut(df["age"], [0, 25, 30, 40, 50, 60, 100], labels=False)
    return df


X = add_feats(X)
Xt = add_feats(Xt)

# label-encoded version for LightGBM
Xl, Xtl = X.copy(), Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], Xt[c]]).astype(str)).categories
    Xl[c] = pd.Categorical(X[c].astype(str), categories=cats).codes
    Xtl[c] = pd.Categorical(Xt[c].astype(str), categories=cats).codes

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X))
oof_c = np.zeros(len(X))
te_l = np.zeros(len(Xt))
te_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=15,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5, random_state=42, verbose=-1)
    m.fit(Xl.iloc[tr], y[tr], eval_set=[(Xl.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(Xl.iloc[va])[:, 1]
    te_l += m.predict_proba(Xtl)[:, 1] / 5

    c = CatBoostClassifier(
        iterations=1500, learning_rate=0.04, depth=6, random_seed=42,
        verbose=0, cat_features=cat_cols, early_stopping_rounds=100,
        thread_count=-1)
    c.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
    oof_c[va] = c.predict_proba(X.iloc[va])[:, 1]
    te_c += c.predict_proba(Xt)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "OOF F1", best_f)

out = pd.DataFrame({"proba": te, "label": (te >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
