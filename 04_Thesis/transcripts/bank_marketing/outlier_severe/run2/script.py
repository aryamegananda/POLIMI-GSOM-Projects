import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].values
X = train.drop(columns=["y"])
Xt = test[X.columns].copy()

cat_cols = [c for c in X.columns if X[c].dtype == object]
num_cols = [c for c in X.columns if c not in cat_cols]


def fe(df):
    df = df.copy()
    df["pdays_flag"] = (df["pdays"] == 999).astype(int)
    df["duration_clip"] = df["duration"].clip(lower=0)
    df["log_duration"] = np.log1p(df["duration_clip"])
    df["campaign_clip"] = df["campaign"].clip(lower=1)
    df["dur_per_call"] = df["duration_clip"] / df["campaign_clip"]
    df["euribor_emp"] = df["euribor3m"] * df["nr.employed"] / 5000.0
    return df


X = fe(X)
Xt = fe(Xt)

# LightGBM matrices: consistent categories
all_df = pd.concat([X, Xt], axis=0, ignore_index=True)
X_l = X.copy()
Xt_l = Xt.copy()
for c in cat_cols:
    cats = sorted(all_df[c].astype(str).unique())
    X_l[c] = pd.Categorical(X[c].astype(str), categories=cats)
    Xt_l[c] = pd.Categorical(Xt[c].astype(str), categories=cats)

# CatBoost matrices: strings
X_c = X.copy()
Xt_c = Xt.copy()
for c in cat_cols:
    X_c[c] = X_c[c].astype(str)
    Xt_c[c] = Xt_c[c].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(len(X))
oof_c = np.zeros(len(X))
te_l = np.zeros(len(Xt))
te_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=1.0, cat_smooth=20,
        random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(X_l.iloc[tr], y[tr])
    oof_l[va] = m.predict_proba(X_l.iloc[va])[:, 1]
    te_l += m.predict_proba(Xt_l)[:, 1] / skf.n_splits

    cb = CatBoostClassifier(
        iterations=600, learning_rate=0.05, depth=6,
        random_seed=SEED, verbose=0, thread_count=-1,
        cat_features=cat_cols)
    cb.fit(X_c.iloc[tr], y[tr])
    oof_c[va] = cb.predict_proba(X_c.iloc[va])[:, 1]
    te_c += cb.predict_proba(Xt_c)[:, 1] / skf.n_splits

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.5, -1
for t in np.arange(0.10, 0.80, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1: %.4f at threshold %.2f" % (best_f, best_t))

out = pd.DataFrame({
    "proba": np.clip(te, 0, 1),
    "label": (te >= best_t).astype(int),
})
out.to_csv("predictions.csv", index=False)
