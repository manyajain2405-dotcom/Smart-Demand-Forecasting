from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.core.config import (
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    SAFE_FEATURE_DATA_PATH,
    SAFE_METRICS_PATH,
    SAFE_MODEL_PATH,
    TARGET_COLUMN,
)
from app.data.features import create_features
from app.data.make_dataset import load_raw_data


# ==========================================
# TECHNIQUE 3: RANDOM FOREST REGRESSION
# ==========================================
def train_test_split_time_series(
    features: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split in timestamp order so future rows never train a past prediction.

    Input: feature rows and a holdout fraction. Output: earlier train rows and
    the latest test rows, matching the order in which forecasting happens.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    ordered = features.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_size))
    if split_index < 1 or split_index >= len(ordered):
        raise ValueError("Not enough feature rows for the requested time split")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Fit a tree ensemble on historical training rows only."""
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit the optional second tree model when its package imports successfully."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ==========================================
# TECHNIQUE 4: MODEL EVALUATION
# ==========================================
def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """Measure errors on unseen rows; lower MAE/RMSE is better, higher R2 is better."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("Actual and predicted values must have the same non-zero length")

    non_zero = actual != 0
    mape = (
        float(np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100)
        if non_zero.any()
        else None
    )
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mape": mape,
    }


def evaluate_regressor(model, test: pd.DataFrame) -> dict[str, float | None]:
    """Predict the chronological holdout using the shared eight-feature contract."""
    actual = test[TARGET_COLUMN]
    predicted = model.predict(test.loc[:, FEATURE_COLUMNS])
    return compute_metrics(actual, predicted)


def run_training_pipeline() -> dict[str, object]:
    """Evaluate candidates and save a new full-data model plus corrected metrics."""
    raw = load_raw_data(RAW_DATA_PATH)
    features = create_features(raw)
    SAFE_FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(SAFE_FEATURE_DATA_PATH, index=False)

    train, test = train_test_split_time_series(features)
    X_train = train.loc[:, FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    baseline_metrics = compute_metrics(test[TARGET_COLUMN], test["lag_1"].to_numpy())
    random_forest = train_random_forest(X_train, y_train)
    random_forest_metrics = evaluate_regressor(random_forest, test)

    xgboost_metrics: dict[str, float | None] | None = None
    xgboost_status = "not installed"
    try:
        import xgboost  # noqa: F401
    except Exception as error:
        xgboost_status = f"unavailable: {type(error).__name__} while loading its native runtime"
    else:
        xgboost_model = train_xgboost(X_train, y_train)
        xgboost_metrics = evaluate_regressor(xgboost_model, test)
        xgboost_status = "evaluated on the same chronological holdout"

    # Refit the deployed Random Forest on all available rows after holdout evaluation.
    final_model = train_random_forest(features.loc[:, FEATURE_COLUMNS], features[TARGET_COLUMN])
    SAFE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, SAFE_MODEL_PATH)

    report: dict[str, object] = {
        "evaluation": {
            "method": "chronological 80/20 holdout",
            "train_size": len(train),
            "test_size": len(test),
            "feature_rows": len(features),
        },
        "baseline": baseline_metrics,
        "random_forest": random_forest_metrics,
        "xgboost": xgboost_metrics,
        "xgboost_status": xgboost_status,
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": {
            name: float(value)
            for name, value in zip(FEATURE_COLUMNS, final_model.feature_importances_)
        },
        "model_artifact": str(SAFE_MODEL_PATH.relative_to(SAFE_MODEL_PATH.parents[1])),
    }
    SAFE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAFE_METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    report = run_training_pipeline()
    print(json.dumps(report, indent=2))
    print(f"Saved leakage-safe model to {SAFE_MODEL_PATH}")
    print(f"Saved corrected evaluation to {SAFE_METRICS_PATH}")


if __name__ == "__main__":
    main()