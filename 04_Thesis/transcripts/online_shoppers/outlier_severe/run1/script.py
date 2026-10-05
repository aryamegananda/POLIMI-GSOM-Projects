import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")

y = train["Revenue"].astype(int).values
Xtr = train.drop(columns=["Revenue"]).copy()
Xte = test.copy()
Xte = Xte[Xtr.columns]

months = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]

def prep(df):
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    df["Month_num"] = df["Month"].map({m: i + 1 for i, m in enumerate(months)}).fillna(0)
    df["Visitor_New"] = (df["VisitorType"] == "New_Visitor").astype(int)
    df["Visitor_Ret"] = (df["VisitorType"] == "Returning_Visitor").astype(int)
    df["Visitor_Other"] = (df["VisitorType"] == "Other").astype(int)
    for m in df["Month"].unique():
        pass
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDur"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                      + df["ProductRelated_Duration"])
    df["DurPerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"].abs() + 1)
    df["PV_log"] = np.sign(df["PageValues"]) * np.log1p(df["PageValues"].abs())
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["Exit_Bounce"] = df["ExitRates"] - df["BounceRates"]
    return df

Ptr, Pte = prep(Xtr), prep(Xte)
month_dummies_cols = sorted(set(Ptr["Month"]) | set(Pte["Month"]))
for m in month_dummies_cols:
    Ptr["M_" + m] = (Ptr["Month"] == m).astype(int)
    Pte["M_" + m] = (Pte["Month"] == m).astype(int)

# numeric matrix
A = Ptr.drop(columns=["Month", "VisitorType"])
B = Pte.drop(columns=["Month", "VisitorType"])[A.columns]

# catboost version with categoricals
Ctr = Ptr.drop(columns=[c for c in Ptr.columns if c.startswith("M_")]).copy()
Cte = Pte.drop(columns=[c for c in Pte.columns if c.startswith("M_")]).copy()
cat_cols = ["Month", "VisitorType"]

def make_models():
    return {
        "lgb": lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
            random_state=SEED, verbose=-1, n_jobs=4),
        "cat": lambda: CatBoostClassifier(
            iterations=600, learning_rate=0.04, depth=6, random_seed=SEED,
            verbose=0, thread_count=4, cat_features=[Ctr.columns.get_loc(c) for c in cat_cols]),
        "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)),
    }

models = make_models()
data = {"lgb": (A, B), "cat": (Ctr, Cte), "lr": (A, B)}
weights = {"lgb": 0.4, "cat": 0.5, "lr": 0.1}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {k: np.zeros(len(y)) for k in models}
for tr, va in skf.split(A, y):
    for k, mk in models.items():
        X = data[k][0]
        m = mk()
        m.fit(X.iloc[tr], y[tr])
        oof[k][va] = m.predict_proba(X.iloc[va])[:, 1]

ens = sum(weights[k] * oof[k] for k in models)
best_t, best_f = 0.5, 0
for t in np.arange(0.15, 0.7, 0.01):
    f = f1_score(y, (ens >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, t
print("CV F1", best_f, "threshold", best_t)

test_pred = np.zeros(len(Xte))
for k, mk in models.items():
    m = mk()
    m.fit(data[k][0], y)
    test_pred += weights[k] * m.predict_proba(data[k][1])[:, 1]

out = pd.DataFrame({"proba": np.clip(test_pred, 0, 1),
                    "label": (test_pred >= best_t).astype(int)})
out.to_csv("predictions.csv", index=False)
