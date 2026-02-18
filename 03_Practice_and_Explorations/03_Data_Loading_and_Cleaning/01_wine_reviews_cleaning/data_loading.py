from pathlib import Path
import pandas as pd

# 1) Load CSV into a DataFrame
def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


# 2) Load JSON into a DataFrame (for later, not needed yet)
def load_json(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_json(file_path)
    return df


# 3) One simple loader that chooses based on file type
def load_data(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return load_csv(file_path)
    elif suffix == ".json":
        return load_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
