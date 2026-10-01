# 0. Import
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# 1. Functions
# 1a. Outliers
def handle_outlier(train,target):
    train = train.copy()
    num_cols = []
    for col in train.select_dtypes(include="number").columns:
        if col != target and col != "ID":
            num_cols.append(col)

    for col in num_cols:
        mask = train[col].notna()
        values = train.loc[mask, [col]]
        model = IsolationForest(contamination="auto", random_state=42)
        model.fit(values)
        pred = model.predict(values)

        outlier_rows = train.index[mask][pred == -1]
        train.loc[outlier_rows, col] = np.nan

    return train
        
    

# # 1b. Missing value
# def clean_missing(train, target):


# # 1c. Handle label noise
# def handle_label_noise(train, target):


# # 1d. Clean
# def clean_all(train, target):