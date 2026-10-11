import pandas as pd, numpy as np
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

def fe(d):
    d = d.copy()
    d["was_contacted"] = (d["pdays"] != 999).astype(int)
    d["pdays"] = d["pdays"].replace(999, np.nan)
    d["age_campaign"] = d["age"] * d["campaign"]
    d["euribor_emp"] = d["euribor3m"] * d["emp.var.rate"]
    return d

X = fe(X); Xt = fe(Xt)

# LightGBM version: category dtype
X_l = X.copy(); Xt_l = Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], Xt[c]])).categories
    X_l[c] = pd.Categorical(X[c], categories=cats)
    Xt_l[c] = pd.Categorical(Xt[c], categories=cats)

# CatBoost version
X_c = X.copy(); Xt_c = Xt.copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X)); oof_c = np.zeros(len(X))
te_l = np.zeros(len(Xt)); te_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=30, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5, cat_smooth=20,
                           random_state=42, verbose=-1, n_jobs=-1)
    m.fit(X_l.iloc[tr], y[tr], eval_set=[(X_l.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(X_l.iloc[va])[:, 1]
    te_l += m.predict_proba(Xt_l)[:, 1] / 5

    cb = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6,
                            random_seed=42, verbose=0, cat_features=cat_cols,
                            early_stopping_rounds=100, thread_count=-1)
    cb.fit(X_c.iloc[tr], y[tr], eval_set=(X_c.iloc[va], y[va]))
    oof_c[va] = cb.predict_proba(X_c.iloc[va])[:, 1]
    te_c += cb.predict_proba(Xt_c)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.7, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1: %.4f at threshold %.3f" % (best_f, best_t))

out = pd.DataFrame({"proba": np.clip(te, 0, 1), "label": (te >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
