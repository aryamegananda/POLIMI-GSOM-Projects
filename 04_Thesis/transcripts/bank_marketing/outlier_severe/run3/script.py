import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]
month_map = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
dow_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}


def prep(df):
    df = df.copy()
    df["age_bad"] = (df["age"] < 16).astype(int)
    df["age"] = df["age"].where(df["age"] >= 16, np.nan)
    df["campaign_neg"] = (df["campaign"] < 0).astype(int)
    df["campaign"] = df["campaign"].abs()
    df["pdays_contacted"] = (df["pdays"] != 999).astype(int)
    df["month_num"] = df["month"].map(month_map)
    df["dow_num"] = df["day_of_week"].map(dow_map)
    df["eur_emp"] = df["euribor3m"] * df["nr.employed"]
    df["emp_per_eur"] = df["nr.employed"] / (df["euribor3m"].abs() + 1)
    return df


train = prep(train)
test = prep(test)
y = train["y"].values
X = train.drop(columns=["y"])
Xt = test[X.columns]

# LightGBM / CatBoost frames
X_l = X.copy()
Xt_l = Xt.copy()
for c in cat_cols:
    cats = pd.Categorical(pd.concat([X[c], Xt[c]]).astype(str)).categories
    X_l[c] = pd.Categorical(X[c].astype(str), categories=cats)
    Xt_l[c] = pd.Categorical(Xt[c].astype(str), categories=cats)

X_c = X.copy()
Xt_c = Xt.copy()
for c in cat_cols:
    X_c[c] = X_c[c].astype(str)
    Xt_c[c] = Xt_c[c].astype(str)

lgb_params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, random_state=42,
                  verbose=-1, n_jobs=-1)
cb_params = dict(iterations=600, learning_rate=0.04, depth=6, l2_leaf_reg=5,
                 random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
for tr, va in skf.split(X, y):
    m1 = lgb.LGBMClassifier(**lgb_params)
    m1.fit(X_l.iloc[tr], y[tr])
    p1 = m1.predict_proba(X_l.iloc[va])[:, 1]
    m2 = CatBoostClassifier(**cb_params)
    m2.fit(X_c.iloc[tr], y[tr])
    p2 = m2.predict_proba(X_c.iloc[va])[:, 1]
    oof[va] = 0.5 * p1 + 0.5 * p2

best_t, best_f = 0.3, 0
for t in np.arange(0.1, 0.7, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("best threshold", best_t, "oof F1", best_f)

m1 = lgb.LGBMClassifier(**lgb_params).fit(X_l, y)
m2 = CatBoostClassifier(**cb_params).fit(X_c, y)
proba = 0.5 * m1.predict_proba(Xt_l)[:, 1] + 0.5 * m2.predict_proba(Xt_c)[:, 1]
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
