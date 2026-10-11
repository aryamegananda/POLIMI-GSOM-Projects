import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].values
n_train = len(train)
full = pd.concat([train.drop(columns=["y"]), test], axis=0, ignore_index=True)

num_cols = ["age", "campaign", "pdays", "previous", "emp.var.rate",
            "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]

# missing indicators
for c in num_cols:
    full[c + "_na"] = full[c].isna().astype(int)
full["na_count"] = full[num_cols].isna().sum(axis=1)

# rule-based fills
m = full["poutcome"] == "nonexistent"
full.loc[m & full["pdays"].isna(), "pdays"] = 999.0
full.loc[m & full["previous"].isna(), "previous"] = 0.0

# iterative imputation using numeric + some encoded categoricals
month_map = {"mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
             "sep": 9, "oct": 10, "nov": 11, "dec": 12}
aux = pd.DataFrame(index=full.index)
aux["month_num"] = full["month"].map(month_map).fillna(6)
aux["dow"] = full["day_of_week"].map({"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}).fillna(3)
aux["pout_s"] = (full["poutcome"] == "success").astype(int)
aux["pout_f"] = (full["poutcome"] == "failure").astype(int)
aux["tel"] = (full["contact"] == "telephone").astype(int)
imp_in = pd.concat([full[num_cols], aux], axis=1)
imp = IterativeImputer(max_iter=8, random_state=SEED, sample_posterior=False)
imp_out = imp.fit_transform(imp_in)
imp_df = pd.DataFrame(imp_out[:, :len(num_cols)], columns=num_cols, index=full.index)
for c in num_cols:
    full[c] = imp_df[c]
full["pdays"] = full["pdays"].clip(0, 999)
full["campaign"] = full["campaign"].clip(lower=1)
full["previous"] = full["previous"].clip(lower=0)

# extra features
full["contacted_before"] = (full["pdays"] < 900).astype(int)
full["euribor_x_nr"] = full["euribor3m"] * full["nr.employed"] / 5000.0
full["age_bin"] = pd.cut(full["age"], [0, 25, 35, 45, 55, 65, 120], labels=False)

feat_cols = [c for c in full.columns]

# LightGBM frame
X_lgb = full[feat_cols].copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype("category")

# CatBoost frame
X_cb = full[feat_cols].copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str)

Xl_tr, Xl_te = X_lgb.iloc[:n_train], X_lgb.iloc[n_train:]
Xc_tr, Xc_te = X_cb.iloc[:n_train], X_cb.iloc[n_train:]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(n_train)
oof_c = np.zeros(n_train)
te_l = np.zeros(len(test))
te_c = np.zeros(len(test))

for tr, va in skf.split(Xl_tr, y):
    ml = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    ml.fit(Xl_tr.iloc[tr], y[tr], eval_set=[(Xl_tr.iloc[va], y[va])],
           eval_metric="auc",
           callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = ml.predict_proba(Xl_tr.iloc[va])[:, 1]
    te_l += ml.predict_proba(Xl_te)[:, 1] / 5

    mc = CatBoostClassifier(
        iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=5,
        random_seed=SEED, verbose=0, eval_metric="AUC",
        cat_features=cat_cols, early_stopping_rounds=100, thread_count=-1)
    mc.fit(Xc_tr.iloc[tr], y[tr], eval_set=(Xc_tr.iloc[va], y[va]))
    oof_c[va] = mc.predict_proba(Xc_tr.iloc[va])[:, 1]
    te_c += mc.predict_proba(Xc_te)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
test_proba = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.3, -1
for t in np.arange(0.05, 0.8, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "OOF F1", best_f)

out = pd.DataFrame({
    "proba": np.clip(test_proba, 0, 1),
    "label": (test_proba >= best_t).astype(int),
})
out.to_csv("predictions.csv", index=False)
