import pandas as pd, numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].astype(int).values
X = train.drop(columns=["y"])
T = test[X.columns].copy()

cat_cols = [c for c in X.columns if X[c].dtype == object]
num_cols = [c for c in X.columns if c not in cat_cols]

def fe(df):
    df = df.copy()
    df["n_missing"] = df[num_cols].isna().sum(axis=1)
    df["never_contacted"] = (df["pdays"] >= 999).astype(float)
    df.loc[df["pdays"].isna(), "never_contacted"] = np.nan
    df["dur_per_camp"] = df["duration"] / (df["campaign"] + 1)
    df["log_dur"] = np.log1p(df["duration"].clip(lower=0))
    return df

X = fe(X); T = fe(T)

# label-encoded version for LightGBM
Xl, Tl = X.copy(), T.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], T[c]]).astype(str)).categories
    Xl[c] = pd.Categorical(X[c].astype(str), categories=cats).codes
    Tl[c] = pd.Categorical(T[c].astype(str), categories=cats).codes

# catboost version
Xc, Tc = X.copy(), T.copy()
for c in cat_cols:
    Xc[c] = Xc[c].astype(str); Tc[c] = Tc[c].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X)); oof_c = np.zeros(len(X))
te_l = np.zeros(len(T)); te_c = np.zeros(len(T))

for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=30, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=2.0,
                           random_state=42, verbose=-1, n_jobs=-1)
    m.fit(Xl.iloc[tr], y[tr], eval_set=[(Xl.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(Xl.iloc[va])[:, 1]
    te_l += m.predict_proba(Tl)[:, 1] / 5

    cb = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6,
                            random_seed=42, verbose=0, thread_count=-1,
                            cat_features=cat_cols, early_stopping_rounds=100)
    cb.fit(Xc.iloc[tr], y[tr], eval_set=(Xc.iloc[va], y[va]))
    oof_c[va] = cb.predict_proba(Xc.iloc[va])[:, 1]
    te_c += cb.predict_proba(Tc)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
proba = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.8, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("CV F1: %.4f at threshold %.2f" % (best_f, best_t))

out = pd.DataFrame({"proba": np.clip(proba, 0, 1),
                    "label": (proba >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
