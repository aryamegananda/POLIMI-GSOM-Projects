from config import RAW_PATH, PROCESSED_PATH, PROCESSED_DIR
from data_loading import load_data
from data_profiling import profile_data
from data_cleaning import clean_wine_reviews

# 1) Orchestrate the workflow: Load → Profile RAW → Clean → Profile CLEAN → Save
def main():
    # 2) Load raw data
    df_raw = load_data(RAW_PATH)

    # 3) Profile raw data (diagnosis)
    profile_data(df_raw, label="RAW")

    # 4) Clean data (apply fixes)
    df_clean = clean_wine_reviews(df_raw)

    # 5) Profile cleaned data (validation)
    profile_data(df_clean, label="CLEANED")

    # 6) Save cleaned dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print("\nSaved cleaned file to:", PROCESSED_PATH)


if __name__ == "__main__":
    main()
