import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]


def prep(df):
    df = df.copy()
    df["age_bad"] = ((df["age"] < 17) | (df["age"] > 100)).astype(int)
    df["age"] = df["age"].clip(17, 100)
    df["campaign"] = df["campaign"].abs().clip(upper=40)
    df["pdays_contacted"] = (df["pdays"] != 999).astype(int)
    df["euribor_emp"] = df["euribor3m"] * df["nr.employed"]
    df["conf_price"] = df["cons.conf.idx"] - df["cons.price.idx"]
    return df


X = prep(train.drop(columns=["y"]))
y = train["y"].values
Xt = prep(test)

# ordinal-coded version (for lgb / xgb)
full = pd.concat([X, Xt], axis=0)
X_ord, Xt_ord = X.copy(), Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(full[c]).categories
    X_ord[c] = pd.Categorical(X[c], categories=cats).codes
    Xt_ord[c] = pd.Categorical(Xt[c], categories=cats).codes

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
names = ["lgb", "cat", "xgb"]
oof = {n: np.zeros(len(X)) for n in names}
tp = {n: np.zeros(len(Xt)) for n in names}

for tr, va in skf.split(X, y):
    # LightGBM
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5, random_state=SEED,
                           verbose=-1, n_jobs=-1)
    m.fit(X_ord.iloc[tr], y[tr], eval_set=[(X_ord.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof["lgb"][va] = m.predict_proba(X_ord.iloc[va])[:, 1]
    tp["lgb"] += m.predict_proba(Xt_ord)[:, 1] / 5

    # CatBoost
    m = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6,
                           random_seed=SEED, verbose=0, cat_features=cat_cols,
                           early_stopping_rounds=100, thread_count=-1)
    m.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
    oof["cat"][va] = m.predict_proba(X.iloc[va])[:, 1]
    tp["cat"] += m.predict_proba(Xt)[:, 1] / 5

    # XGBoost
    m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.02, max_depth=4,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                          reg_lambda=5, random_state=SEED, n_jobs=-1,
                          eval_metric="logloss", early_stopping_rounds=100,
                          tree_method="hist")
    m.fit(X_ord.iloc[tr], y[tr], eval_set=[(X_ord.iloc[va], y[va])], verbose=False)
    oof["xgb"][va] = m.predict_proba(X_ord.iloc[va])[:, 1]
    tp["xgb"] += m.predict_proba(Xt_ord)[:, 1] / 5

oof_blend = np.mean([oof[n] for n in names], axis=0)
test_blend = np.mean([tp[n] for n in names], axis=0)

best_t, best_f = 0.5, -1
for t in np.arange(0.1, 0.8, 0.005):
    f = f1_score(y, (oof_blend >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

out = pd.DataFrame({"proba": np.clip(test_blend, 0, 1),
                    "label": (test_blend >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
