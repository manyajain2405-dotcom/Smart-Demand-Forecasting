# Smart Demand Forecasting

A Streamlit application for exploring the project's supplied hourly energy-demand data, training a leakage-safe forecast model, and inspecting future demand scenarios.

## Problem Statement

Energy demand changes over time. A forecasting workflow can help reveal recurring patterns, estimate near-term demand, and make model performance easier to inspect. This project demonstrates that workflow using the demand observations already included in `data/raw/energy.csv`.

## Objectives

- Clean and validate timestamped demand observations.
- Build calendar, lag, and rolling features from past observations only.
- Compare a previous-hour baseline with a Random Forest using a chronological holdout.
- Report MAE, RMSE, R2, and MAPE, and expose feature importance.
- Explore historical demand, peak hours, and future what-if scenarios in a dashboard.

## Technologies

Python, pandas, NumPy, scikit-learn, joblib, Plotly, Streamlit, and pytest. XGBoost is an optional comparison model.

## Machine Learning Techniques

- **Data preprocessing:** parse timestamps, coerce demand to numeric values, remove invalid rows and duplicate timestamps, and sort chronologically.
- **Feature engineering:** hour, weekday, month, weekend flag, previous-hour demand, previous-day demand, and the previous 24 observations' mean and standard deviation.
- **Time-series split:** the earliest 80% of engineered rows are used for training; the latest 20% are held out for evaluation.
- **Baseline:** predict the next hour using the latest observed hour (`lag_1`).
- **Random Forest Regression:** an ensemble of decision trees trained on the time-ordered training rows. The deployed model is refit on all available rows only after holdout evaluation.
- **Optional XGBoost:** evaluated on the same holdout when the package and its native runtime are available.
- **Evaluation:** MAE and RMSE quantify error magnitude; R2 compares the predictions with a constant-mean prediction; MAPE reports percentage error for non-zero actual values.

The rolling statistics are shifted by one hour before they are computed, so the target hour is never included in its own features. The original `data/processed/features.csv`, `models/trained_model.pkl`, and `reports/metrics.json` are preserved as historical project artifacts. The original model was serialized with scikit-learn 1.6.1; loading it under a different version can emit a compatibility warning. The dashboard prefers the separately trained leakage-safe model when it exists.

## Measured Evaluation

The corrected pipeline was run on the repository's existing dataset. It produced 476 feature rows and used a chronological split of 380 training rows and 96 holdout rows.

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Previous-hour baseline | 6.403 | 7.818 | 0.731 |
| Random Forest | 4.626 | 5.723 | 0.856 |

These are the results written to `reports/corrected_metrics.json`; they are not the older scores in `reports/metrics.json`. XGBoost was not evaluated in the current macOS environment because its native `libomp.dylib` runtime is absent. No XGBoost score is claimed.

## Project Structure

```text
smart-demand-forecasting/
├── app/
│   ├── core/config.py
│   ├── data/features.py
│   ├── data/make_dataset.py
│   └── models/
│       ├── forecast.py
│       └── train.py
├── dashboard/dashboard_app.py
├── data/
│   ├── raw/energy.csv
│   └── processed/              # original and leakage-safe feature data
├── models/                     # original and leakage-safe model artifacts
├── reports/                    # original and corrected evaluation reports
├── tests/
├── requirements.txt
├── README.md
└── VIVA_EXPLANATION.md
```

## Installation (macOS)

Python 3.10 or newer is recommended. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

XGBoost is optional. To try the comparison on macOS, install its Python package and the OpenMP runtime:

```bash
python -m pip install xgboost
brew install libomp
```

Then rerun training. Homebrew is not needed for the dashboard or Random Forest model.

## Run the Application

Train and evaluate the leakage-safe model (this writes new files and does not replace the original artifacts):

```bash
python -m app.models.train
```

Start the dashboard:

```bash
streamlit run dashboard/dashboard_app.py
```

Run the test suite:

```bash
python -m pytest -q
```

The dashboard includes Overview, Consumption, Forecast, and Model insights views. Forecast horizons can be set from 1 to 168 hours; the demand scenario scales recursive model predictions and can be downloaded as CSV. This is a stress-test control, not a causal simulation of weather, price, or other external drivers.

## Model Evaluation

`app/models/train.py` evaluates models only on the latest chronological holdout. The Random Forest and XGBoost candidates use the same training rows and holdout. The previous-hour baseline provides a simple reference. After evaluation, a separate Random Forest is trained on all feature rows for future forecasting. Its feature importances are available in the dashboard and corrected JSON report.

## Screenshots

Add screenshots of the Overview, Consumption, and Forecast views here after launching the local dashboard.

## Future Enhancements

- Add weather, tariff, and calendar event inputs when suitable data is available.
- Extend the historical dataset to cover multiple seasons and years.
- Add prediction intervals and rolling-origin backtesting.
- Compare optional models such as XGBoost after its native runtime is available.
- Add accessibility and deployment checks for a hosted dashboard.