import numpy as np
import pandas as pd
import pytest

from app.models.train import compute_metrics, train_test_split_time_series


def test_time_split_keeps_latest_rows_out_of_training() -> None:
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="h"),
            "demand": np.arange(10),
        }
    )

    train, test = train_test_split_time_series(features, test_size=0.2)

    assert len(train) == 8
    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()


def test_metric_calculation_reports_mae_rmse_and_r2() -> None:
    metrics = compute_metrics(np.array([1.0, 3.0, 5.0]), np.array([2.0, 2.0, 4.0]))

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.0))
    assert metrics["r2"] == pytest.approx(0.625)


def test_time_split_rejects_invalid_fraction() -> None:
    features = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=2, freq="h")})

    with pytest.raises(ValueError, match="between 0 and 1"):
        train_test_split_time_series(features, test_size=1)