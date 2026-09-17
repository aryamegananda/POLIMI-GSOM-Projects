# 0. Import
import pandas as pd
from sqlalchemy import create_engine
from config import DB_URL


# 1. Create database connection
def get_engine():
    engine = create_engine(DB_URL)
    return engine


# 2. Load it to DB
def load_to_db(df):
    engine = get_engine()

    df.to_sql(
        name = "test_results",
        con = engine,
        if_exists = "replace",
        index = False
    )

    print(f"Loaded {len(df)} rows into PostgreSQL table 'test_results'")


# 3. Check
def verify_load():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM test_results LIMIT 5", engine)
    print(f"\n--- Verification (first 5 rows from database) ---")
    print(df)

    count = pd.read_sql("SELECT COUNT(*) as total FROM test_results", engine)
    print(f"\nTotal rows in database: {count['total'][0]}")


# 4. Run
if __name__ == "__main__":
    clean_data = pd.read_csv("data/clean/cleaned_results.csv")
    load_to_db(clean_data)
    verify_load()