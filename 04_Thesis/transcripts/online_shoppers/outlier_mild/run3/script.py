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
n_train = len(train)
full = pd.concat([train.drop(columns=["Revenue"]), test], axis=0, ignore_index=True)


def make_features(df):
    df = df.copy()
    num_cols = ["Administrative", "Administrative_Duration", "Informational",
                "Informational_Duration", "ProductRelated", "ProductRelated_Duration"]
    # some values are negative (noisy); keep raw and add abs versions
    for c in num_cols:
        df[c + "_abs"] = df[c].abs()
    df["TotalPages"] = (df["Administrative_abs"] + df["Informational_abs"]
                        + df["ProductRelated_abs"])
    df["TotalDuration"] = (df["Administrative_Duration_abs"] + df["Informational_Duration_abs"]
                           + df["ProductRelated_Duration_abs"])
    df["DurPerPage"] = df["TotalDuration"] / (df["TotalPages"] + 1)
    df["ProdDurPerPage"] = df["ProductRelated_Duration_abs"] / (df["ProductRelated_abs"] + 1)
    df["PV_pos"] = (df["PageValues"] > 0).astype(int)
    df["PV_abs"] = df["PageValues"].abs()
    df["PV_log"] = np.log1p(df["PV_abs"])
    df["Exit_minus_Bounce"] = df["ExitRates"] - df["BounceRates"]
    df["Exit_x_PV"] = df["ExitRates"].abs() * df["PV_abs"]
    month_map = {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8,
                 "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df["Month_num"] = df["Month"].map(month_map).fillna(0)
    df = pd.get_dummies(df, columns=["Month", "VisitorType"], dtype=int)
    return df


X_all = make_features(full)
X_all.columns = [str(c).replace(" ", "_") for c in X_all.columns]
X = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_lgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)
test_lgb = np.zeros(len(X_test))
test_cb = np.zeros(len(X_test))

for tr, va in skf.split(X, y):
    m1 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                            min_child_samples=20, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.7, reg_lambda=1.0,
                            random_state=SEED, verbose=-1, n_jobs=-1)
    m1.fit(X.iloc[tr], y[tr])
    oof_lgb[va] = m1.predict_proba(X.iloc[va])[:, 1]
    test_lgb += m1.predict_proba(X_test)[:, 1] / skf.n_splits

    m2 = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
                            random_seed=SEED, verbose=0, thread_count=-1)
    m2.fit(X.iloc[tr], y[tr])
    oof_cb[va] = m2.predict_proba(X.iloc[va])[:, 1]
    test_cb += m2.predict_proba(X_test)[:, 1] / skf.n_splits

oof = 0.5 * oof_lgb + 0.5 * oof_cb
test_proba = 0.5 * test_lgb + 0.5 * test_cb

best_t, best_f1 = 0.5, -1
for t in np.arange(0.15, 0.75, 0.01):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f1:
        best_f1, best_t = f, t
print("OOF best threshold", best_t, "F1", best_f1)

labels = (test_proba >= best_t).astype(int)
pd.DataFrame({"proba": np.clip(test_proba, 0, 1), "label": labels}).to_csv(
    "predictions.csv", index=False)
