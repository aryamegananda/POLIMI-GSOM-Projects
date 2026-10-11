import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

tr = pd.read_csv("train.csv")
te = pd.read_csv("test_features.csv")
target = "y"
cats = ["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]

def prep(df):
    d = df.copy()
    d["age"] = d["age"].where(d["age"] > 0, np.nan)
    d["campaign"] = d["campaign"].abs()
    d["pdays_flag"] = (d["pdays"] != 999).astype(int)
    d["pdays"] = d["pdays"].where(d["pdays"] != 999, -1)
    d["emp_euribor"] = d["emp.var.rate"] * d["euribor3m"]
    d["conf_minus_price"] = d["cons.conf.idx"] - d["cons.price.idx"]
    return d

X = prep(tr.drop(columns=[target]))
y = tr[target].values
Xt = prep(te)

# LightGBM version: category dtype
full = pd.concat([X, Xt], axis=0, keys=["a", "b"])
Xl_all = full.copy()
for c in cats:
    Xl_all[c] = Xl_all[c].astype("category")
Xl, Xlt = Xl_all.loc["a"], Xl_all.loc["b"]

# CatBoost version: string categoricals
Xc, Xct = X.copy(), Xt.copy()
for c in cats:
    Xc[c] = Xc[c].astype(str)
    Xct[c] = Xct[c].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X)); oof_c = np.zeros(len(X))
pt_l = np.zeros(len(Xt)); pt_c = np.zeros(len(Xt))

for tri, vai in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                           min_child_samples=30, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5, random_state=42,
                           verbose=-1, n_jobs=-1)
    m.fit(Xl.iloc[tri], y[tri])
    oof_l[vai] = m.predict_proba(Xl.iloc[vai])[:, 1]
    pt_l += m.predict_proba(Xlt)[:, 1] / 5

    c = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                           random_seed=42, verbose=0, cat_features=cats,
                           thread_count=-1)
    c.fit(Xc.iloc[tri], y[tri])
    oof_c[vai] = c.predict_proba(Xc.iloc[vai])[:, 1]
    pt_c += c.predict_proba(Xct)[:, 1] / 5

oof = 0.5 * oof_l + 0.5 * oof_c
pt = 0.5 * pt_l + 0.5 * pt_c

best_t, best_f = 0.5, 0
for t in np.arange(0.1, 0.7, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF F1", best_f, "threshold", best_t)

out = pd.DataFrame({"proba": np.clip(pt, 0, 1),
                    "label": (pt >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
