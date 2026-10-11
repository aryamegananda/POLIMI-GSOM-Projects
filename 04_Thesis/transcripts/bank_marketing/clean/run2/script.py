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
Xt = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == object]


def fe(df):
    df = df.copy()
    df["pdays_never"] = (df["pdays"] == 999).astype(int)
    df["pdays_clean"] = df["pdays"].replace(999, np.nan)
    df["n_unknown"] = (df[cat_cols] == "unknown").sum(axis=1)
    df["euri_x_emp"] = df["euribor3m"] * df["emp.var.rate"]
    return df


X = fe(X)
Xt = fe(Xt)

# Ordinal-encoded version for LightGBM
X_l = X.copy()
Xt_l = Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], Xt[c]])).categories
    X_l[c] = pd.Categorical(X[c], categories=cats).codes
    Xt_l[c] = pd.Categorical(Xt[c], categories=cats).codes

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(len(X))
oof_c = np.zeros(len(X))
te_l = np.zeros(len(Xt))
te_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=15,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5, random_state=SEED,
        verbose=-1, n_jobs=-1)
    m.fit(X_l.iloc[tr], y[tr], eval_set=[(X_l.iloc[va], y[va])],
          eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(X_l.iloc[va])[:, 1]
    te_l += m.predict_proba(Xt_l)[:, 1] / skf.n_splits

    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=5,
        random_seed=SEED, eval_metric="AUC", od_type="Iter", od_wait=100,
        verbose=0, thread_count=-1)
    cb.fit(X.iloc[tr], y[tr], cat_features=cat_cols,
           eval_set=(X.iloc[va], y[va]), use_best_model=True)
    oof_c[va] = cb.predict_proba(X.iloc[va])[:, 1]
    te_c += cb.predict_proba(Xt)[:, 1] / skf.n_splits

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.7, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1 %.4f at threshold %.3f" % (best_f, best_t))

out = pd.DataFrame({"proba": np.clip(te, 0, 1),
                    "label": (te >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
