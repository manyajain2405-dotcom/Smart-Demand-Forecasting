from pathlib import Path

import pandas as pd

from app.core.config import RAW_DATA_PATH, SAFE_CLEAN_DATA_PATH
from app.data.features import clean_data


# ==========================================
# TECHNIQUE 1: DATA PREPROCESSING
# ==========================================
def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the source CSV; input is a path, output is a dataframe."""
    if not path.exists():
        raise FileNotFoundError(f"Raw demand data not found at {path}")
    return pd.read_csv(path)


def prepare_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Clean the source dataset without changing the committed clean.csv file."""
    return clean_data(load_raw_data(path))


def main() -> None:
    cleaned = prepare_dataset()
    SAFE_CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(SAFE_CLEAN_DATA_PATH, index=False)
    print(f"Saved {len(cleaned)} cleaned rows to {SAFE_CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()