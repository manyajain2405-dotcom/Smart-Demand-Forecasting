from __future__ import annotations

import pandas as pd

from app.core.config import FEATURE_COLUMNS, TARGET_COLUMN


# ==========================================
# TECHNIQUE 1: DATA PREPROCESSING
# ==========================================
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, numeric, unique timestamp/demand rows.

    Input: raw rows with timestamp and demand columns.
    Output: clean rows; invalid timestamps or demand values are omitted.
    """
    required = {"timestamp", TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = data.loc[:, ["timestamp", TARGET_COLUMN]].copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp", TARGET_COLUMN])
    cleaned = cleaned.sort_values("timestamp").drop_duplicates(
        subset="timestamp", keep="last"
    )
    cleaned = cleaned.reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No valid timestamp/demand rows remain after cleaning.")
    return cleaned


# ==========================================
# TECHNIQUE 2: FEATURE ENGINEERING
# ==========================================
def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build calendar, lag, and trailing-window features without target leakage.

    Input: clean, chronologically ordered hourly demand rows.
    Output: rows with eight model inputs and their next-step training target.
    """
    cleaned = clean_data(data)
    timestamps = cleaned["timestamp"]
    prior_demand = cleaned[TARGET_COLUMN].shift(1)

    features = pd.DataFrame({"timestamp": timestamps, TARGET_COLUMN: cleaned[TARGET_COLUMN]})
    features["hour"] = timestamps.dt.hour
    features["day_of_week"] = timestamps.dt.day_of_week
    features["month"] = timestamps.dt.month
    features["is_weekend"] = timestamps.dt.day_of_week.isin([5, 6]).astype(int)
    features["lag_1"] = cleaned[TARGET_COLUMN].shift(1)
    features["lag_24"] = cleaned[TARGET_COLUMN].shift(24)

    # Shift first so each rolling window contains only demand known before the target hour.
    features["rolling_mean_24"] = prior_demand.rolling(window=24, min_periods=24).mean()
    features["rolling_std_24"] = prior_demand.rolling(window=24, min_periods=24).std()

    features = features.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    validate_feature_columns(features)
    return features


def validate_feature_columns(data: pd.DataFrame) -> None:
    """Raise a clear error if engineered model inputs are incomplete."""
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if TARGET_COLUMN not in data.columns:
        missing.append(TARGET_COLUMN)
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")