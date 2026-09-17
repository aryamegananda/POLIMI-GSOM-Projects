# 0. Import
import os

# 1. Define connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "eurofins_lab",
    "user": "postgres",
    "password": os.getenv("DB_PASSWORD")
}

# 2. SQL Alchemy connection string
DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['passowrd']}@{DB_CONFIG['host']:{DB_CONFIG['port']}/{DB_CONFIG['database']}}"

# 3. File path
RAW_DATA_DIR = "data/raw"
CLEAN_DATA_DIR = "data/clean"

# 4. Global schema
SCHEMA = {
    "sample_id": "str", # format: SITE-YYYY-NNNN
    "test_type": "str", # from controlled list
    "result_value": "float", # numeric measurement
    "unit": "str", # SI units and standarized units
    "run_date": "datetime", # UTC timestamp
    "operator_id": "str", # operator who runs the test / PIC
    "site_code": "str", # VIM, POG, MIL
    "status": "str" # passed / failed / pending
}

# 5. Valid values mapping
VALID_SITES = ["VIM", "POG", "MIL"]
VALID_STATUSES = ["passed", "failed", "pending"]
VALID_TEST_TYPES = ["pH", "endotoxin", "sterility", "bioburden", "conductivity", "TOC", "microbial_limit", "potency"]
VALID_UNITS = ["pH", "EU/mL", "pass/fail", "CFU/mL", "us/cm", "ppb", "CFU/g", "mg/mL"]