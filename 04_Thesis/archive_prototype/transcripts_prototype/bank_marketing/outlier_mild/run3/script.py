import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].values
X = train.drop(columns=["y"])
Xt = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == object]

def clean(df):
    df = df.copy()
    df.loc[df["age"] < 16, "age"] = np.nan
    df.loc[df["duration"] < 0, "duration"] = np.nan
    df.loc[df["campaign"] < 1, "campaign"] = np.nan
    df["pdays_flag"] = (df["pdays"] == 999).astype(int)
    df["log_dur"] = np.log1p(df["duration"].clip(lower=0))
    df["dur_per_camp"] = df["duration"] / df["campaign"]
    df["euri_emp"] = df["euribor3m"] * df["nr.employed"] / 5000
    return df

X = clean(X)
Xt = clean(Xt)

# categorical handling
full = pd.concat([X, Xt], axis=0)
X_lgb, Xt_lgb = X.copy(), Xt.copy()
for c in cat_cols:
    cats = sorted(full[c].astype(str).unique())
    X_lgb[c] = pd.Categorical(X[c].astype(str), categories=cats)
    Xt_lgb[c] = pd.Categorical(Xt[c].astype(str), categories=cats)

X_cb, Xt_cb = X.copy(), Xt.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str)
    Xt_cb[c] = Xt_cb[c].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(len(X))
oof_c = np.zeros(len(X))
te_l = np.zeros(len(Xt))
te_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED,
        verbose=-1, n_jobs=-1)
    m.fit(X_lgb.iloc[tr], y[tr],
          eval_set=[(X_lgb.iloc[va], y[va])],
          eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(X_lgb.iloc[va])[:, 1]
    te_l += m.predict_proba(Xt_lgb)[:, 1] / skf.n_splits

    c = CatBoostClassifier(
        iterations=1500, learning_rate=0.05, depth=6,
        random_seed=SEED, verbose=0, cat_features=cat_cols,
        eval_metric="AUC", early_stopping_rounds=100, thread_count=-1)
    c.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]))
    oof_c[va] = c.predict_proba(X_cb.iloc[va])[:, 1]
    te_c += c.predict_proba(Xt_cb)[:, 1] / skf.n_splits

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.5, -1
for t in np.arange(0.1, 0.9, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "OOF F1", best_f)

out = pd.DataFrame({"proba": np.clip(te, 0, 1),
                    "label": (te >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
