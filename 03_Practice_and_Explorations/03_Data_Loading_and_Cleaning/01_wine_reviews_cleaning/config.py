from pathlib import Path

# 1) Base project folders
# This makes paths work no matter where your project is on your computer.
BASE_DIR = Path(__file__).resolve().parents[1]   # project root (one level above /src)
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 2) Choose which raw file to process (start with ONE file)
RAW_FILENAME = "winemag-data-130k-v2.csv"
RAW_PATH = RAW_DIR / RAW_FILENAME

# 3) Output (cleaned) file path
PROCESSED_FILENAME = "wine_reviews_clean.csv"
PROCESSED_PATH = PROCESSED_DIR / PROCESSED_FILENAME