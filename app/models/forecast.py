from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import FEATURE_COLUMNS, TARGET_COLUMN
from app.data.features import clean_data


# ==========================================
# TECHNIQUE 5: DEMAND PREDICTION
# ==========================================
def forecast_future(
    model: object,
    history: pd.DataFrame,
    steps: int,
    demand_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Forecast future hours recursively, using only observations available so far.

    Input: fitted regressor, historical timestamp/demand rows, horizon, and scenario scale.
    Output: timestamped baseline and what-if predictions for each future hour.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(demand_multiplier) or demand_multiplier <= 0:
        raise ValueError("demand_multiplier must be a positive finite number")

    cleaned = clean_data(history)
    if len(cleaned) < 24:
        raise ValueError("At least 24 valid hourly observations are required to forecast.")

    known_demand = cleaned[TARGET_COLUMN].astype(float).tolist()
    last_timestamp = cleaned["timestamp"].iloc[-1]
    predictions: list[dict[str, object]] = []

    for offset in range(1, steps + 1):
        timestamp = last_timestamp + pd.Timedelta(hours=offset)
        recent = np.asarray(known_demand[-24:], dtype=float)
        previous = known_demand[-1]
        row = pd.DataFrame(
            [
                {
                    "hour": timestamp.hour,
                    "day_of_week": timestamp.day_of_week,
                    "month": timestamp.month,
                    "is_weekend": int(timestamp.day_of_week in (5, 6)),
                    "lag_1": previous,
                    "lag_24": known_demand[-24],
                    "rolling_mean_24": float(recent.mean()),
                    "rolling_std_24": float(recent.std(ddof=1)),
                }
            ],
            columns=FEATURE_COLUMNS,
        )
        baseline = float(model.predict(row)[0])  # type: ignore[attr-defined]
        scenario = baseline * demand_multiplier
        predictions.append(
            {
                "timestamp": timestamp,
                "baseline_demand": baseline,
                "scenario_demand": scenario,
            }
        )
        known_demand.append(scenario)

    return pd.DataFrame(predictions)