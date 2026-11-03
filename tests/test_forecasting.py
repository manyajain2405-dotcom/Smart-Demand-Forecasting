import numpy as np
import pandas as pd
import pytest

from app.core.config import FEATURE_COLUMNS
from app.data.features import create_features
from app.models.forecast import forecast_future


class PersistenceModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["lag_1"].to_numpy()


def sample_history(rows: int = 72) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "demand": np.arange(1, rows + 1, dtype=float),
        }
    )


def test_rolling_features_use_only_observations_before_target() -> None:
    features = create_features(sample_history())
    first = features.iloc[0]

    assert first["demand"] == 25
    assert first["lag_1"] == 24
    assert first["lag_24"] == 1
    assert first["rolling_mean_24"] == pytest.approx(np.mean(np.arange(1, 25)))
    assert list(features.loc[:, FEATURE_COLUMNS].columns) == FEATURE_COLUMNS


def test_feature_builder_sorts_and_removes_duplicate_timestamps() -> None:
    history = sample_history()
    history = pd.concat([history.iloc[[1, 0]], history.iloc[2:], history.iloc[[2]]])

    features = create_features(history)

    assert features["timestamp"].is_monotonic_increasing
    assert features["timestamp"].is_unique


def test_recursive_forecast_returns_scenario_and_baseline() -> None:
    forecast = forecast_future(PersistenceModel(), sample_history(), 3, 1.1)

    assert forecast["baseline_demand"].tolist() == [72.0, 79.2, 87.12]
    assert forecast["scenario_demand"].tolist() == pytest.approx([79.2, 87.12, 95.832])
    assert forecast["timestamp"].iloc[0] == pd.Timestamp("2020-01-04 00:00:00")


def test_forecast_requires_sufficient_history() -> None:
    with pytest.raises(ValueError, match="24 valid hourly observations"):
        forecast_future(PersistenceModel(), sample_history(23), 1)