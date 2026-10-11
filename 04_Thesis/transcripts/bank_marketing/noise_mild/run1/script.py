import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
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
    df["pdays_flag"] = (df["pdays"] == 999).astype(int)
    df["pdays_c"] = df["pdays"].where(df["pdays"] != 999, -1)
    df["euribor_emp"] = df["euribor3m"] * df["emp.var.rate"]
    df["nr_per_eur"] = df["nr.employed"] / (df["euribor3m"] + 1)
    df["campaign_log"] = np.log1p(df["campaign"])
    return df

X = fe(X)
Xt = fe(Xt)

# Label/category version for LightGBM
all_df = pd.concat([X, Xt], axis=0)
X_lgb = X.copy()
Xt_lgb = Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(all_df[c]).categories
    X_lgb[c] = pd.Categorical(X[c], categories=cats)
    Xt_lgb[c] = pd.Categorical(Xt[c], categories=cats)

# One-hot version for XGBoost
all_oh = pd.get_dummies(all_df, columns=cat_cols, dtype=float)
X_oh = all_oh.iloc[:len(X)].reset_index(drop=True)
Xt_oh = all_oh.iloc[len(X):].reset_index(drop=True)

# CatBoost version
X_cb = X.copy()
Xt_cb = Xt.copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {k: np.zeros(len(X)) for k in ["lgb", "xgb", "cb"]}
tp = {k: np.zeros(len(Xt)) for k in ["lgb", "xgb", "cb"]}

for tr, va in skf.split(X, y):
    # LightGBM
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5.0, cat_smooth=20,
                           random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(X_lgb.iloc[tr], y[tr], eval_set=[(X_lgb.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof["lgb"][va] = m.predict_proba(X_lgb.iloc[va])[:, 1]
    tp["lgb"] += m.predict_proba(Xt_lgb)[:, 1] / 5

    # XGBoost
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                          reg_lambda=5.0, random_state=SEED, n_jobs=-1,
                          eval_metric="logloss", early_stopping_rounds=100,
                          tree_method="hist")
    m.fit(X_oh.iloc[tr], y[tr], eval_set=[(X_oh.iloc[va], y[va])], verbose=False)
    oof["xgb"][va] = m.predict_proba(X_oh.iloc[va])[:, 1]
    tp["xgb"] += m.predict_proba(Xt_oh)[:, 1] / 5

    # CatBoost
    m = CatBoostClassifier(iterations=800, learning_rate=0.04, depth=6,
                           l2_leaf_reg=5, random_seed=SEED, verbose=0,
                           cat_features=cat_cols, early_stopping_rounds=100,
                           thread_count=-1)
    m.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]))
    oof["cb"][va] = m.predict_proba(X_cb.iloc[va])[:, 1]
    tp["cb"] += m.predict_proba(Xt_cb)[:, 1] / 5

def blend(d):
    return (d["lgb"] + d["xgb"] + d["cb"]) / 3

oof_b = blend(oof)
test_b = blend(tp)

best_t, best_f = 0.3, 0
for t in np.arange(0.1, 0.7, 0.005):
    f = f1_score(y, (oof_b >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF best threshold", best_t, "F1", best_f)

out = pd.DataFrame({"proba": np.clip(test_b, 0, 1),
                    "label": (test_b >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
