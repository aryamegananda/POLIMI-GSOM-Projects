import pandas as pd, numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["y"].values
X = train.drop(columns=["y"])
Xt = test[X.columns].copy()

cat_cols = [c for c in X.columns if X[c].dtype == object]
months = {m: i for i, m in enumerate(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], 1)}
days = {"mon":1,"tue":2,"wed":3,"thu":4,"fri":5}

def fe(d):
    d = d.copy()
    d["pdays_flag"] = (d["pdays"] == 999).astype(int)
    d["month_num"] = d["month"].map(months)
    d["dow_num"] = d["day_of_week"].map(days)
    d["prev_contact"] = (d["previous"] > 0).astype(int)
    d["rate_x_emp"] = d["euribor3m"] * d["nr.employed"] / 5000.0
    return d

X = fe(X); Xt = fe(Xt)

# LightGBM frame
all_cat = pd.concat([X[cat_cols], Xt[cat_cols]])
Xl, Xtl = X.copy(), Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(all_cat[c]).categories
    Xl[c] = pd.Categorical(X[c], categories=cats)
    Xtl[c] = pd.Categorical(Xt[c], categories=cats)

lgb_params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                  random_state=42, verbose=-1, n_jobs=-1)
cb_params = dict(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0,
                 cat_features=cat_cols, thread_count=-1)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof_l = np.zeros(len(X)); oof_c = np.zeros(len(X))
for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**lgb_params).fit(Xl.iloc[tr], y[tr])
    oof_l[va] = m.predict_proba(Xl.iloc[va])[:, 1]
    c = CatBoostClassifier(**cb_params).fit(X.iloc[tr], y[tr])
    oof_c[va] = c.predict_proba(X.iloc[va])[:, 1]

oof = 0.5 * oof_l + 0.5 * oof_c
best_t, best_f = 0.3, 0
for t in np.arange(0.1, 0.7, 0.005):
    f = f1_score(y, (oof > t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("CV F1", best_f, "threshold", best_t)

ml = lgb.LGBMClassifier(**lgb_params).fit(Xl, y)
mc = CatBoostClassifier(**cb_params).fit(X, y)
proba = 0.5 * ml.predict_proba(Xtl)[:, 1] + 0.5 * mc.predict_proba(Xt)[:, 1]
label = (proba > best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
