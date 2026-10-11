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

y = train["y"].astype(int).values
X = train.drop(columns=["y"])
Xt = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == "object"]

def fe(df):
    df = df.copy()
    df["pdays_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)
    df["dur_log"] = np.log1p(df["duration"])
    df["dur_per_camp"] = df["duration"] / (df["campaign"] + 1)
    df["euribor_emp"] = df["euribor3m"] * df["nr.employed"] / 5000.0
    return df

X = fe(X)
Xt = fe(Xt)

# LightGBM frame with category dtype
def lgb_frame(df, ref):
    d = df.copy()
    for c in cat_cols:
        d[c] = pd.Categorical(d[c], categories=sorted(ref[c].unique()))
    return d

Xl = lgb_frame(X, X)
Xtl = lgb_frame(Xt, X)

# CatBoost frame
Xc = X.copy()
Xtc = Xt.copy()

lgb_params = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
    random_state=SEED, verbose=-1, n_jobs=-1,
)
cb_params = dict(
    iterations=600, learning_rate=0.05, depth=6, random_seed=SEED,
    verbose=0, cat_features=cat_cols, thread_count=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
pred_test = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m1 = lgb.LGBMClassifier(**lgb_params)
    m1.fit(Xl.iloc[tr], y[tr])
    p1v = m1.predict_proba(Xl.iloc[va])[:, 1]
    p1t = m1.predict_proba(Xtl)[:, 1]

    m2 = CatBoostClassifier(**cb_params)
    m2.fit(Xc.iloc[tr], y[tr])
    p2v = m2.predict_proba(Xc.iloc[va])[:, 1]
    p2t = m2.predict_proba(Xtc)[:, 1]

    oof[va] = 0.5 * p1v + 0.5 * p2v
    pred_test += (0.5 * p1t + 0.5 * p2t) / skf.get_n_splits()

# threshold optimization on OOF
best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({
    "proba": np.clip(pred_test, 0, 1),
    "label": (pred_test >= best_t).astype(int),
})
out.to_csv("predictions.csv", index=False)
