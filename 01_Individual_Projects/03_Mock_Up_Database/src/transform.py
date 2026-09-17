# 0. Import
import pandas as pd
import numpy as np
from config import SCHEMA, VALID_SITES, VALID_STATUSES, VALID_TEST_TYPES, VALID_UNITS


# 1. Column mappings
COLUMN_MAP = {
    "vimodrone": {
        "SampleID": "sample_id",
        "TestType": "test_type",
        "Result": "result_value",
        "Unit": "unit",
        "test_date":"run_date",
        "Operator":"operator_id",
        "Status": "status"
    },
    "poggibonsi": {
        "cod_campione":"sample_id",
        "tipo_test": "test_type",
        "valore": "result_value",
        "unita": "unit",
        "data_analisi":"run_date",
        "operatore":"operator_id",
        "esito": "status"
    },
    "milan_hq": {
        "Sample_Code": "sample_id",
        "Test": "test_type",
        "Value": "result_value",
        "UNIT": "unit",
        "Date": "run_date",
        "Tech": "operator_id",
        "Result_Status": "status",
    }
}


# 2. Status mapping
STATUS_MAP = {
    "passato": "passed",
    "fallito": "failed",
    "in attesa": "pending",
    "pass": "passed",
    "fail": "failed",
    "failed": "failed",
    "pending": "pending",
    "passed": "passed"
}


# 3. Test types mapping
# 4. Units mapping