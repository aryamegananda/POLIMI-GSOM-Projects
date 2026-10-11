import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
train_x = train.drop(columns=["Revenue"])
test_x = test.copy()

full = pd.concat([train_x, test_x], axis=0, ignore_index=True)

# Feature engineering
full["Month"] = full["Month"].astype(str)
full["VisitorType"] = full["VisitorType"].astype(str)
month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jul": 7,
             "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
full["MonthNum"] = full["Month"].map(month_map).fillna(0)

full["TotalPages"] = (full["Administrative"].abs() + full["Informational"].abs()
                      + full["ProductRelated"].abs())
full["TotalDuration"] = (full["Administrative_Duration"].abs() + full["Informational_Duration"].abs()
                         + full["ProductRelated_Duration"].abs())
full["DurPerPage"] = full["TotalDuration"] / (full["TotalPages"] + 1)
full["PR_DurPerPage"] = full["ProductRelated_Duration"].abs() / (full["ProductRelated"].abs() + 1)
full["PageValues_x_Exit"] = full["PageValues"] * (1 - full["ExitRates"])
full["LogPageValues"] = np.sign(full["PageValues"]) * np.log1p(full["PageValues"].abs())
full["HasPageValue"] = (full["PageValues"] > 0).astype(int)
full["Bounce_Exit"] = full["BounceRates"] + full["ExitRates"]

full = pd.get_dummies(full, columns=["Month", "VisitorType"], dtype=int)
full = full.astype(float)

n = len(train_x)
X = full.iloc[:n].reset_index(drop=True)
Xt = full.iloc[n:].reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(n)
pred = np.zeros(len(Xt))

for tr, va in skf.split(X, y):
    m1 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                            min_child_samples=30, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.7, reg_lambda=2.0,
                            random_state=SEED, verbose=-1)
    m1.fit(X.iloc[tr], y[tr])
    m2 = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.04,
                            random_seed=SEED, verbose=0, thread_count=-1)
    m2.fit(X.iloc[tr], y[tr])
    p_va = 0.5 * m1.predict_proba(X.iloc[va])[:, 1] + 0.5 * m2.predict_proba(X.iloc[va])[:, 1]
    oof[va] = p_va
    pred += (0.5 * m1.predict_proba(Xt)[:, 1] + 0.5 * m2.predict_proba(Xt)[:, 1]) / skf.n_splits

# Threshold optimization on OOF
best_t, best_f = 0.5, -1
for t in np.arange(0.1, 0.8, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("OOF best F1: %.4f at threshold %.2f" % (best_f, best_t))

out = pd.DataFrame({"proba": np.clip(pred, 0, 1),
                    "label": (pred >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
