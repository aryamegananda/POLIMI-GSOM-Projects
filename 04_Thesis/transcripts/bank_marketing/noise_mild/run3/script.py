import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].values
X = train.drop(columns=["y"])
Xt = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == object]
month_map = {m: i for i, m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], 1)}


def fe(d):
    d = d.copy()
    d["pdays_contacted"] = (d["pdays"] != 999).astype(int)
    d["pdays"] = d["pdays"].replace(999, -1)
    d["month_num"] = d["month"].map(month_map)
    d["euribor_x_emp"] = d["euribor3m"] * d["emp.var.rate"]
    d["unknown_count"] = (d[cat_cols] == "unknown").sum(axis=1)
    return d


X = fe(X)
Xt = fe(Xt)

# LightGBM inputs: category dtype
Xl, Xtl = X.copy(), Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], Xt[c]])).categories
    Xl[c] = pd.Categorical(X[c], categories=cats)
    Xtl[c] = pd.Categorical(Xt[c], categories=cats)

lgb_params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, cat_smooth=20,
                  random_state=42, verbose=-1, n_jobs=-1)
cb_params = dict(iterations=600, learning_rate=0.04, depth=6,
                 random_seed=42, verbose=0, thread_count=-1)


def fit_cb(Xa, ya):
    m = CatBoostClassifier(**cb_params, cat_features=cat_cols)
    m.fit(Xa, ya)
    return m


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
for tr, va in skf.split(X, y):
    m1 = lgb.LGBMClassifier(**lgb_params)
    m1.fit(Xl.iloc[tr], y[tr])
    p1 = m1.predict_proba(Xl.iloc[va])[:, 1]
    m2 = fit_cb(X.iloc[tr], y[tr])
    p2 = m2.predict_proba(X.iloc[va])[:, 1]
    oof[va] = 0.5 * p1 + 0.5 * p2

best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF best F1", best_f, "threshold", best_t)

m1 = lgb.LGBMClassifier(**lgb_params).fit(Xl, y)
m2 = fit_cb(X, y)
proba = 0.5 * m1.predict_proba(Xtl)[:, 1] + 0.5 * m2.predict_proba(Xt)[:, 1]
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
