# 0. Import
import os
import sys
import time
import platform
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import cleaning


# 1. Config
SEED = 42
PIPELINE = "manual"
RESULT_FILE = "results/manual_results.csv"

DATASET = {
    "bank_marketing": {"target": "y"},
    "online_shoppers": {"target": "Revenue"},
    "credit_card": {"target": "default.payment.next.month"}
}

CONDITIONS = [
    "clean", "missing_mild", "missing_severe", "outlier_mild", "outlier_severe", 
    "noise_mild" , "noise_severe"
]

# 2. Feature Engineering
