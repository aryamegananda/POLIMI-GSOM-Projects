import pandas as pd

# 1) Basic profile function (reusable for any dataset)
def profile_data(df: pd.DataFrame, label: str, top_n_missing: int = 10) -> None:
    print("\n==============================")
    print(f"PROFILE: {label}")
    print("==============================")

    # 2) Size of the dataset
    print("Shape:", df.shape)

    # 3) Column names
    print("\nColumns:")
    print(list(df.columns))

    # 4) Data types
    print("\nData Types:")
    print(df.dtypes)

    # 5) Missing values (top N)
    print(f"\nMissing Values (top {top_n_missing}):")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing.head(top_n_missing))

    # 6) Duplicates
    print("\nNumber of duplicated rows:")
    print(df.duplicated().sum())
