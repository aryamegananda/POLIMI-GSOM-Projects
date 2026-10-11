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
X_all = pd.concat([train.drop(columns=["y"]), test], ignore_index=True)

cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]
macro = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

# Rule-based fills
non_ex = X_all["poutcome"] == "nonexistent"
X_all.loc[non_ex & X_all["pdays"].isna(), "pdays"] = 999.0
X_all.loc[non_ex & X_all["previous"].isna(), "previous"] = 0.0
succ = X_all["poutcome"] == "success"
# pdays != 999 implies previous contact
X_all.loc[(X_all["pdays"] < 999) & X_all["previous"].isna(), "previous"] = 1.0

# Feature engineering
month_map = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
dow_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}
X_all["month_num"] = X_all["month"].map(month_map).astype(float)
X_all["dow_num"] = X_all["day_of_week"].map(dow_map).astype(float)
X_all["n_missing"] = X_all[["age", "campaign", "pdays", "previous"] + macro].isna().sum(axis=1)

# Impute macro columns (iterative, using month and other macro vars)
imp_cols = macro + ["month_num"]
orig_min = X_all[macro].min()
orig_max = X_all[macro].max()
imp = IterativeImputer(max_iter=10, random_state=SEED, sample_posterior=False)
imputed = imp.fit_transform(X_all[imp_cols])
imputed = pd.DataFrame(imputed, columns=imp_cols)
for c in macro:
    X_all[c + "_imp"] = imputed[c].clip(orig_min[c], orig_max[c]).values
# keep macro_imp as primary; drop raw macro
for c in macro:
    X_all[c] = X_all[c + "_imp"]
    X_all.drop(columns=[c + "_imp"], inplace=True)

X_all["prev_contacted"] = (X_all["pdays"] < 999).astype(float)
X_all.loc[X_all["pdays"].isna(), "prev_contacted"] = np.nan
X_all["euribor_x_emp"] = X_all["euribor3m"] * X_all["emp.var.rate"]

# LightGBM frame
X_lgb = X_all.copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype("category")
# CatBoost frame
X_cb = X_all.copy()
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
    m = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(Xl_tr.iloc[tr], y[tr], eval_set=[(Xl_tr.iloc[va], y[va])],
          eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[va] = m.predict_proba(Xl_tr.iloc[va])[:, 1]
    te_l += m.predict_proba(Xl_te)[:, 1] / 5

    c = CatBoostClassifier(
        iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=5,
        eval_metric="AUC", random_seed=SEED, verbose=0, od_type="Iter",
        od_wait=100, cat_features=cat_cols, thread_count=-1)
    c.fit(Xc_tr.iloc[tr], y[tr], eval_set=(Xc_tr.iloc[va], y[va]), use_best_model=True)
    oof_c[va] = c.predict_proba(Xc_tr.iloc[va])[:, 1]
    te_c += c.predict_proba(Xc_te)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
te = 0.5 * te_l + 0.5 * te_c

best_t, best_f = 0.3, -1
for t in np.arange(0.05, 0.75, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1: %.4f at threshold %.3f" % (best_f, best_t))

pred = pd.DataFrame({"proba": np.clip(te, 0, 1),
                     "label": (te >= best_t).astype(int)})
pred.to_csv("predictions.csv", index=False)
