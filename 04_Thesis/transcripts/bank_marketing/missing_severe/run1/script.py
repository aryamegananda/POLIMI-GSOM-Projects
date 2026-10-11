import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
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

# Feature engineering
month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
             "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
dow_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}
full["month_num"] = full["month"].map(month_map)
full["dow_num"] = full["day_of_week"].map(dow_map)
full["contact_num"] = (full["contact"] == "cellular").astype(int)
full["pdays_missing_flag"] = full["pdays"].isna().astype(int)
full["pdays_never"] = (full["pdays"] == 999).astype(float)
full.loc[full["pdays"].isna(), "pdays_never"] = np.nan
full["poutcome_num"] = full["poutcome"].map({"nonexistent": 0, "failure": 1, "success": 2})

# Imputed versions of numerics (helps recover macro variables)
imp_in = full[num_cols + ["month_num", "dow_num", "contact_num", "poutcome_num"]].copy()
imp = IterativeImputer(max_iter=5, random_state=SEED, sample_posterior=False)
imp_out = imp.fit_transform(imp_in)
for i, c in enumerate(num_cols):
    full[c + "_imp"] = imp_out[:, i]

full["n_missing"] = full[num_cols].isna().sum(axis=1)

# LightGBM matrix
X_lgb = full.copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype("category")
# CatBoost matrix
X_cb = full.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str)

Xl_tr, Xl_te = X_lgb.iloc[:n_train], X_lgb.iloc[n_train:]
Xc_tr, Xc_te = X_cb.iloc[:n_train], X_cb.iloc[n_train:]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(n_train)
oof_c = np.zeros(n_train)
te_l = np.zeros(len(test))
te_c = np.zeros(len(test))

for fold, (tr, va) in enumerate(skf.split(Xl_tr, y)):
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(Xl_tr.iloc[tr], y[tr], eval_set=[(Xl_tr.iloc[va], y[va])],
          eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(Xl_tr.iloc[va])[:, 1]
    te_l += m.predict_proba(Xl_te)[:, 1] / 5

    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=5,
        eval_metric="AUC", random_seed=SEED, verbose=False,
        early_stopping_rounds=100, cat_features=cat_cols, thread_count=-1)
    cb.fit(Xc_tr.iloc[tr], y[tr], eval_set=(Xc_tr.iloc[va], y[va]))
    oof_c[va] = cb.predict_proba(Xc_tr.iloc[va])[:, 1]
    te_c += cb.predict_proba(Xc_te)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
test_proba = 0.5 * te_l + 0.5 * te_c

best_thr, best_f1 = 0.5, -1
for t in np.arange(0.1, 0.71, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f1:
        best_f1, best_thr = f, t
print("OOF F1: %.4f at threshold %.2f" % (best_f1, best_thr))

out = pd.DataFrame({
    "proba": np.clip(test_proba, 0, 1),
    "label": (test_proba >= best_thr).astype(int),
})
out.to_csv("predictions.csv", index=False)
