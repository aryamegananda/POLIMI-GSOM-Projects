import pandas as pd

# 1) Cleaning function specific to the Wine Reviews dataset
# But written in a style you can reuse for other datasets.
def clean_wine_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 2) Drop duplicates (safe default)
    df = df.drop_duplicates()

    # 3) Drop unnecessary columns
    # "Unnamed: 0" is usually an index artifact from saving a CSV.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # "taster_twitter_handle" has high missing and low value for most analysis.
    if "taster_twitter_handle" in df.columns:
        df = df.drop(columns=["taster_twitter_handle"])

    # 4) Handle missing categorical columns
    # Fill with "Unknown" + add indicator columns (missingness can be informative)
    for col in ["region_1", "region_2", "designation"]:
        if col in df.columns:
            df[f"has_{col}"] = df[col].notna().astype(int)
            df[col] = df[col].fillna("Unknown")

    # 5) Keep taster_name (don’t impute names), but add indicator
    if "taster_name" in df.columns:
        df["has_taster_name"] = df["taster_name"].notna().astype(int)

    # 6) Price imputation: median (robust to outliers / skew)
    if "price" in df.columns:
        median_price = df["price"].median()
        df["price"] = df["price"].fillna(median_price)

    # 7) Clean whitespace in text columns (common real-world mess)
    text_columns = [
        "country", "province", "region_1", "region_2",
        "variety", "winery", "taster_name", "title", "designation"
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # 8) Drop duplicates again AFTER dropping columns/filling values
    # Cleaning steps can make previously different rows become identical.
    df = df.drop_duplicates()

    return df
