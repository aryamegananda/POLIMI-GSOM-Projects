# 0. Import
import time
from extract import extract_all
from transform import transform_all
from load import load_to_db, verify_load

# 1. Function
def run_pipeline():
    print("EUROFINS ETL PIPELINE")

    start = time.time()

    # a. Extract
    print(f"\n[1/3] EXTRACTING...")
    raw_data = extract_all()

    # b. Transform
    print(f"\n[2/3] TRANSFORMING...")
    clean_data = transform_all(raw_data)

    # c. Load
    print(f"\n[3/3] LOADING...")
    load_to_db(clean_data)

    # d. Done
    elapsed = round(time.time() - start, 2)
    print(f"PIPELINE COMPLETED - {elapsed} seconds")

    # e. Verify
    verify_load()


# 2. Run
if __name__ == "__main__":
    run_pipeline()