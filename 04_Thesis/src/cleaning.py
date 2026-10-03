# 0. Import
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from cleanlab.filter import find_label_issues


# 1. Functions
# 1a. Outliers Handling
VALIDITY_RULES = {
    "bank_marketing": {
        "age":      (17, 100),
        "duration": (0, None),
        "campaign": (1, None),
        "pdays":    (0, 999),
        "previous": (0, None),
    },
    "online_shoppers": {
        "Administrative": (0, None),
        "Informational": (0, None),
        "ProductRelated": (0, None),
        "Administrative_Duration": (0, None),
        "Informational_Duration": (0, None),
        "ProductRelated_Duration": (0, None),
        "BounceRates": (0, 1),
        "ExitRates": (0, 1),
        "PageValues": (0, None),
        "SpecialDay": (0, 1),
        "OperatingSystems": (1, None),
        "Browser": (1, None),
        "Region": (1, None),
        "TrafficType": (1, None),
    },
    "credit_card": {
        "LIMIT_BAL": (0, None),
        "SEX": (1, 2),
        "EDUCATION": (0, 6),
        "MARRIAGE": (0, 3),
        "AGE": (18, 100),
        "PAY_0": (-2, 9),
        "PAY_2": (-2, 9),
        "PAY_3": (-2, 9),
        "PAY_4": (-2, 9),
        "PAY_5": (-2, 9),
        "PAY_6": (-2, 9),
        "PAY_AMT1": (0, None),
        "PAY_AMT2": (0, None),
        "PAY_AMT3": (0, None),
        "PAY_AMT4": (0, None),
        "PAY_AMT5": (0, None),
        "PAY_AMT6": (0, None),
    },
}

def validity_check(train, rules):
    train = train.copy()
    for col in rules:
        low, high = rules[col]
        if low is not None:
            train.loc[train[col] < low, col] = np.nan
        if high is not None:
            train.loc[train[col] > high, col] = np.nan
    return train
    

# 1b. Missing value
def clean_missing(train, target):
    train = train.copy()

    # numeric columns (except target) and categorical columns
    num_cols = []
    cat_cols = []
    for col in train.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and not pd.api.types.is_bool_dtype(train[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    # nothing missing -> nothing to do
    if train[num_cols].isnull().sum().sum() == 0:
        return train

    # categoricals as one-hot, so MICE can use them as predictors
    # (needed because PuckTrick removes all numeric values of a row together)
    X = pd.get_dummies(train[num_cols + cat_cols], columns=cat_cols, dtype=float)

    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=train.index)

    # only the numeric columns are replaced; categoricals stay as they were
    train[num_cols] = X_imputed[num_cols]
    return train


# 1c. Handle label noise
def handle_label_noise(train, target):
    train = train.copy()

    X = pd.get_dummies(train.drop(columns=[target]), dtype=float)
    y = train[target].values

    # out-of-sample probabilities from a simple, fixed model (5-fold CV)
    model = HistGradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

    # True = label probably wrong
    issues = find_label_issues(labels=y, pred_probs=pred_probs)

    train = train[~issues].reset_index(drop=True)
    return train


# 1d. Clean
def clean_all(train, target, rules):
    report = {}
    report["rows_before"] = len(train)
    nan_before = train.isnull().sum().sum()

    train = validity_check(train, rules)
    report["values_invalid"] = int(train.isnull().sum().sum() - nan_before)
    report["values_imputed"] = int(train.isnull().sum().sum())

    train = clean_missing(train, target)

    rows_before_noise = len(train)
    train = handle_label_noise(train, target)
    report["rows_removed_label_noise"] = rows_before_noise - len(train)
    report["rows_after"] = len(train)

    return train, report