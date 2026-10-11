import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
Xtr = train.drop(columns=["Revenue"])
Xte = test.copy()
if "Revenue" in Xte.columns:
    Xte = Xte.drop(columns=["Revenue"])

n_train = len(Xtr)
full = pd.concat([Xtr, Xte], axis=0, ignore_index=True)


def fe(df):
    df = df.copy()
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                           + df["ProductRelated_Duration"])
    df["DurPerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"].abs() + 1)
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["PV_log"] = np.sign(df["PageValues"]) * np.log1p(df["PageValues"].abs())
    df["ExitBounceDiff"] = df["ExitRates"] - df["BounceRates"]
    df["ExitBounceMean"] = (df["ExitRates"] + df["BounceRates"]) / 2
    df["PR_log"] = np.sign(df["ProductRelated"]) * np.log1p(df["ProductRelated"].abs())
    df["PRD_log"] = np.sign(df["ProductRelated_Duration"]) * np.log1p(df["ProductRelated_Duration"].abs())
    return df


full = fe(full)
full = pd.get_dummies(full, columns=["Month", "VisitorType"], dtype=int)
full.columns = [str(c).replace(" ", "_") for c in full.columns]

X = full.iloc[:n_train].reset_index(drop=True)
Xt = full.iloc[n_train:].reset_index(drop=True)


def make_models():
    return {
        "lgb": LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=42, verbose=-1, n_jobs=-1),
        "cat": CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                                  random_seed=42, verbose=0, thread_count=-1),
        "hgb": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                              max_leaf_nodes=15, l2_regularization=1.0,
                                              random_state=42),
    }


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
names = list(make_models().keys())
oof = {n: np.zeros(n_train) for n in names}

for tr_idx, va_idx in skf.split(X, y):
    models = make_models()
    for n, m in models.items():
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof[n][va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]

oof_blend = np.mean([oof[n] for n in names], axis=0)

best_t, best_f1 = 0.5, -1
for t in np.arange(0.15, 0.75, 0.01):
    f = f1_score(y, (oof_blend >= t).astype(int))
    if f > best_f1:
        best_f1, best_t = f, t
print("OOF F1: %.4f at threshold %.2f" % (best_f1, best_t))

preds = []
for n, m in make_models().items():
    m.fit(X, y)
    preds.append(m.predict_proba(Xt)[:, 1])
proba = np.mean(preds, axis=0)
label = (proba >= best_t).astype(int)

pd.DataFrame({"proba": proba, "label": label}).to_csv("predictions.csv", index=False)
