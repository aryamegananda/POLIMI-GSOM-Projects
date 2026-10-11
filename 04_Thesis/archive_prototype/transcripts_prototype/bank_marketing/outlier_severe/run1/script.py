import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

target = "y"
y = train[target].values
X = train.drop(columns=[target]).copy()
Xt = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == object]


def fe(df):
    df = df.copy()
    df["pdays_none"] = (df["pdays"] == 999).astype(int)
    df["pdays_clean"] = df["pdays"].where(df["pdays"] != 999, -1)
    df["dur_clip"] = df["duration"].clip(lower=0)
    df["log_dur"] = np.log1p(df["dur_clip"])
    df["campaign_abs"] = df["campaign"].abs()
    df["age_clip"] = df["age"].clip(18, 100)
    df["dur_per_campaign"] = df["dur_clip"] / (df["campaign_abs"] + 1)
    df["euribor_x_emp"] = df["euribor3m"] * df["emp.var.rate"]
    df["month_day"] = df["month"] + "_" + df["day_of_week"]
    df["job_edu"] = df["job"] + "_" + df["education"]
    return df


X = fe(X)
Xt = fe(Xt)
cat_cols = [c for c in X.columns if X[c].dtype == object]

# LightGBM version with category dtype
Xl, Xtl = X.copy(), Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], Xt[c]]).astype(str)).categories
    Xl[c] = pd.Categorical(X[c].astype(str), categories=cats)
    Xtl[c] = pd.Categorical(Xt[c].astype(str), categories=cats)

# CatBoost version
Xc, Xtc = X.copy(), Xt.copy()
for c in cat_cols:
    Xc[c] = Xc[c].astype(str)
    Xtc[c] = Xtc[c].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(len(X))
oof_c = np.zeros(len(X))
test_l = np.zeros(len(Xt))
test_c = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=1.0, cat_smooth=20,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(Xl.iloc[tr], y[tr])
    oof_l[va] = m.predict_proba(Xl.iloc[va])[:, 1]
    test_l += m.predict_proba(Xtl)[:, 1] / skf.n_splits

    cb = CatBoostClassifier(
        iterations=600, learning_rate=0.05, depth=6,
        random_seed=SEED, verbose=0, thread_count=-1)
    cb.fit(Xc.iloc[tr], y[tr], cat_features=cat_cols)
    oof_c[va] = cb.predict_proba(Xc.iloc[va])[:, 1]
    test_c += cb.predict_proba(Xtc)[:, 1] / skf.n_splits

oof = 0.5 * oof_l + 0.5 * oof_c
proba = 0.5 * test_l + 0.5 * test_c

best_t, best_f = 0.5, -1
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("Best threshold", best_t, "OOF F1", best_f)

pred = (proba >= best_t).astype(int)
pd.DataFrame({"proba": np.clip(proba, 0, 1), "label": pred}).to_csv("predictions.csv", index=False)
