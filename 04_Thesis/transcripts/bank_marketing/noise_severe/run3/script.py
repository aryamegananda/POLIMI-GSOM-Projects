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
X = train.drop(columns=["y"]).copy()
Xt = test.copy()[X.columns]

cat_cols = [c for c in X.columns if X[c].dtype == object]

def fe(df):
    df = df.copy()
    df["no_prev_contact"] = (df["pdays"] == 999).astype(int)
    df["pdays_clean"] = df["pdays"].replace(999, -1)
    df["log_duration"] = np.log1p(df["duration"])
    df["dur_per_campaign"] = df["duration"] / (df["campaign"] + 1)
    df["euribor_x_emp"] = df["euribor3m"] * df["emp.var.rate"]
    return df

X = fe(X)
Xt = fe(Xt)

# LightGBM frame (category dtype)
Xl, Xtl = X.copy(), Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], Xt[c]])).categories
    Xl[c] = pd.Categorical(X[c], categories=cats)
    Xtl[c] = pd.Categorical(Xt[c], categories=cats)

# CatBoost frame
Xc, Xtc = X.copy(), Xt.copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(len(X)); oof_c = np.zeros(len(X))
pt_l = np.zeros(len(Xt)); pt_c = np.zeros(len(Xt))

lgb_params = dict(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED,
                  verbose=-1, n_jobs=-1, cat_smooth=20)

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(Xl.iloc[tr], y[tr], eval_set=[(Xl.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(Xl.iloc[va])[:, 1]
    pt_l += m.predict_proba(Xtl)[:, 1] / 5

    c = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6,
                           random_seed=SEED, verbose=0, thread_count=-1,
                           cat_features=cat_cols, early_stopping_rounds=100)
    c.fit(Xc.iloc[tr], y[tr], eval_set=(Xc.iloc[va], y[va]))
    oof_c[va] = c.predict_proba(Xc.iloc[va])[:, 1]
    pt_c += c.predict_proba(Xtc)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
pt = 0.5 * pt_l + 0.5 * pt_c

best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.005):
    f = f1_score(y, (oof > t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(pt, 0, 1),
                    "label": (pt > best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
