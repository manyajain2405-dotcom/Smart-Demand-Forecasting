"""Paths and feature names shared by the data, model, and dashboard modules."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "energy.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "clean.csv"
SAFE_CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "clean_leakage_safe.csv"
FEATURE_DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"
SAFE_FEATURE_DATA_PATH = BASE_DIR / "data" / "processed" / "features_leakage_safe.csv"
LEGACY_MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"
SAFE_MODEL_PATH = BASE_DIR / "models" / "trained_model_leakage_safe.pkl"
LEGACY_METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
SAFE_METRICS_PATH = BASE_DIR / "reports" / "corrected_metrics.json"
FIGURES_PATH = BASE_DIR / "reports" / "figures"

TARGET_COLUMN = "demand"
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "rolling_mean_24",
    "rolling_std_24",
]