from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import (
    FEATURE_COLUMNS,
    LEGACY_METRICS_PATH,
    LEGACY_MODEL_PATH,
    RAW_DATA_PATH,
    SAFE_METRICS_PATH,
    SAFE_MODEL_PATH,
    TARGET_COLUMN,
)
from app.data.features import clean_data
from app.models.forecast import forecast_future


COLORS = {
    "background": "#08111F",
    "surface": "#101D30",
    "blue": "#3B82F6",
    "cyan": "#36D6E7",
    "green": "#A3E635",
    "orange": "#F59E0B",
    "text": "#ECF4FF",
    "muted": "#91A4BD",
}


@st.cache_data
def load_history() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Demand data not found at {RAW_DATA_PATH}")
    return clean_data(pd.read_csv(RAW_DATA_PATH))


@st.cache_resource
def load_model() -> tuple[object, str]:
    if SAFE_MODEL_PATH.exists():
        return joblib.load(SAFE_MODEL_PATH), "Leakage-safe retrained model"
    if LEGACY_MODEL_PATH.exists():
        return joblib.load(LEGACY_MODEL_PATH), "Preserved historical model"
    raise FileNotFoundError("No trained model found. Run `python -m app.models.train` first.")


@st.cache_data
def load_metrics() -> tuple[dict[str, object] | None, str]:
    if SAFE_METRICS_PATH.exists():
        return json.loads(SAFE_METRICS_PATH.read_text()), "Corrected chronological evaluation"
    if LEGACY_METRICS_PATH.exists():
        return json.loads(LEGACY_METRICS_PATH.read_text()), "Original saved evaluation"
    return None, "No evaluation report found"


def style_chart(figure: go.Figure, height: int = 360) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": COLORS["text"], "size": 12},
        margin={"l": 8, "r": 12, "t": 34, "b": 8},
        height=height,
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"gridcolor": "rgba(145,164,189,0.12)", "zerolinecolor": "rgba(145,164,189,0.16)"},
        yaxis={"gridcolor": "rgba(145,164,189,0.12)", "zerolinecolor": "rgba(145,164,189,0.16)"},
    )
    return figure


def metric_value(metrics: dict[str, object] | None, model_name: str, key: str) -> float | None:
    if not metrics:
        return None
    section = metrics.get(model_name)
    if not isinstance(section, dict):
        return None
    value = section.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def render_overview(history: pd.DataFrame, metrics: dict[str, object] | None) -> None:
    demand = history[TARGET_COLUMN]
    mean_demand = float(demand.mean())
    peak_index = demand.idxmax()
    peak_row = history.loc[peak_index]
    mae = metric_value(metrics, "random_forest", "mae")
    latest = history["timestamp"].max()

    st.markdown('<div class="section-kicker">SYSTEM SUMMARY / LIVE DATASET</div>', unsafe_allow_html=True)
    st.subheader("Demand at a glance")
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Mean hourly demand", f"{mean_demand:,.1f}", delta="demand units")
    kpi_columns[1].metric("Observed peak", f"{peak_row[TARGET_COLUMN]:,.1f}", delta=peak_row["timestamp"].strftime("%b %d, %H:%M"))
    kpi_columns[2].metric("History available", f"{len(history):,} hours", delta=f"through {latest:%b %d, %Y}")
    kpi_columns[3].metric("Holdout MAE", f"{mae:.2f}" if mae is not None else "n/a", delta="lower is better")

    st.markdown('<div class="section-kicker">RECENT LOAD</div>', unsafe_allow_html=True)
    recent = history.tail(24 * 7)
    figure = px.area(
        recent,
        x="timestamp",
        y=TARGET_COLUMN,
        color_discrete_sequence=[COLORS["blue"]],
        labels={"timestamp": "Hour", TARGET_COLUMN: "Demand"},
    )
    figure.update_traces(line={"width": 2.2}, fillcolor="rgba(59,130,246,0.14)")
    st.plotly_chart(style_chart(figure, 390), width="stretch")

    first, second = st.columns([1.3, 0.7])
    with first:
        st.markdown("#### Model snapshot")
        forest_mae = metric_value(metrics, "random_forest", "mae")
        baseline_mae = metric_value(metrics, "baseline", "mae")
        if forest_mae is not None and baseline_mae is not None:
            delta = baseline_mae - forest_mae
            st.write(
                f"The Random Forest holdout MAE is **{forest_mae:.2f}**, compared with "
                f"**{baseline_mae:.2f}** for the previous-hour baseline "
                f"({delta:+.2f} demand units on this evaluation split)."
            )
        else:
            st.write("Evaluation metrics are not available. Run the training command to create corrected scores.")
    with second:
        st.markdown("#### Coverage")
        st.write(f"**From**  {history['timestamp'].min():%b %d, %Y}")
        st.write(f"**To**  {latest:%b %d, %Y}")
        st.write(f"**Rows**  {len(history):,} hourly observations")


def render_consumption(history: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">HISTORICAL ANALYSIS</div>', unsafe_allow_html=True)
    st.subheader("Consumption patterns")
    start_default = history["timestamp"].min().date()
    end_default = history["timestamp"].max().date()
    selected_dates = st.date_input(
        "Date window",
        value=(start_default, end_default),
        min_value=start_default,
        max_value=end_default,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start, end = selected_dates
        selected = history.loc[
            history["timestamp"].dt.date.between(start, end)
        ].copy()
    else:
        selected = history.copy()
    if selected.empty:
        st.info("No observations are available for this date window.")
        return

    daily = selected.set_index("timestamp")[TARGET_COLUMN].resample("D").sum().rename("Demand").reset_index()
    weekly = selected.set_index("timestamp")[TARGET_COLUMN].resample("W-SUN").mean().rename("Average demand").reset_index()
    monthly = selected.set_index("timestamp")[TARGET_COLUMN].resample("MS").mean().rename("Average demand").reset_index()

    daily_col, weekly_col = st.columns(2)
    with daily_col:
        st.markdown("#### Daily demand")
        figure = px.bar(daily, x="timestamp", y="Demand", color_discrete_sequence=[COLORS["blue"]])
        st.plotly_chart(style_chart(figure, 320), width="stretch")
    with weekly_col:
        st.markdown("#### Weekly average")
        figure = px.line(weekly, x="timestamp", y="Average demand", markers=True, color_discrete_sequence=[COLORS["cyan"]])
        st.plotly_chart(style_chart(figure, 320), width="stretch")

    monthly_col, peak_col = st.columns([1.1, 0.9])
    with monthly_col:
        st.markdown("#### Monthly average")
        figure = px.bar(monthly, x="timestamp", y="Average demand", color_discrete_sequence=[COLORS["green"]])
        st.plotly_chart(style_chart(figure, 300), width="stretch")
    with peak_col:
        st.markdown("#### Highest-demand hours")
        peaks = selected.nlargest(8, TARGET_COLUMN).sort_values(TARGET_COLUMN)
        figure = px.bar(
            peaks,
            x=TARGET_COLUMN,
            y="timestamp",
            orientation="h",
            color=TARGET_COLUMN,
            color_continuous_scale=[COLORS["blue"], COLORS["cyan"], COLORS["green"]],
            labels={TARGET_COLUMN: "Demand", "timestamp": "Hour"},
        )
        st.plotly_chart(style_chart(figure, 300), width="stretch")

    st.markdown("#### Average demand by hour and weekday")
    heatmap = selected.assign(
        weekday=selected["timestamp"].dt.day_of_week,
        hour=selected["timestamp"].dt.hour,
    ).pivot_table(index="weekday", columns="hour", values=TARGET_COLUMN, aggfunc="mean")
    heatmap = heatmap.reindex(index=range(7), columns=range(24))
    heatmap.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    figure = px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale=[[0, "#102038"], [0.5, COLORS["blue"]], [1, COLORS["cyan"]]],
        labels={"x": "Hour of day", "y": "Weekday", "color": "Demand"},
    )
    st.plotly_chart(style_chart(figure, 330), width="stretch")


def render_forecast(history: pd.DataFrame, model: object) -> None:
    st.markdown('<div class="section-kicker">FORWARD LOOK / WHAT-IF</div>', unsafe_allow_html=True)
    st.subheader("Forecast horizon")
    horizon_col, scenario_col = st.columns([1, 1])
    with horizon_col:
        steps = st.slider("Hours ahead", min_value=1, max_value=168, value=24, step=1)
    with scenario_col:
        scale_percent = st.slider("Demand scenario", min_value=80, max_value=120, value=100, step=1)
    multiplier = scale_percent / 100

    result = forecast_future(model, history, steps, demand_multiplier=multiplier)
    mean_forecast = result["scenario_demand"].mean()
    peak_forecast = result["scenario_demand"].max()
    kpi_columns = st.columns(3)
    kpi_columns[0].metric("Mean scenario demand", f"{mean_forecast:,.1f}")
    kpi_columns[1].metric("Scenario peak", f"{peak_forecast:,.1f}")
    kpi_columns[2].metric("Scenario scale", f"{scale_percent}%", delta=f"{scale_percent - 100:+d}% from baseline")

    recent = history.tail(min(72, len(history)))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=recent["timestamp"], y=recent[TARGET_COLUMN], name="Observed", line={"color": COLORS["muted"], "width": 1.6}))
    figure.add_trace(go.Scatter(x=result["timestamp"], y=result["baseline_demand"], name="Model baseline", line={"color": COLORS["blue"], "width": 2, "dash": "dot"}))
    figure.add_trace(go.Scatter(x=result["timestamp"], y=result["scenario_demand"], name="What-if scenario", line={"color": COLORS["cyan"], "width": 2.5}, fill="tonexty", fillcolor="rgba(54,214,231,0.08)"))
    st.plotly_chart(style_chart(figure, 430), width="stretch")

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast CSV",
        data=csv,
        file_name="demand_forecast.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    st.caption("The scenario scales each recursive prediction and feeds it into the next hour. It is a demand stress test, not a causal estimate of weather or pricing effects.")


def render_model_insights(
    model: object,
    metrics: dict[str, object] | None,
    model_label: str,
    metrics_label: str,
) -> None:
    st.markdown('<div class="section-kicker">MODEL CARD</div>', unsafe_allow_html=True)
    st.subheader("Evaluation and drivers")
    st.caption(f"Forecast model: {model_label}  ·  Scores: {metrics_label}")

    rows = []
    for key, label in (("baseline", "Previous-hour baseline"), ("random_forest", "Random Forest"), ("xgboost", "XGBoost")):
        values = metrics.get(key) if metrics else None
        if isinstance(values, dict):
            rows.append(
                {
                    "Model": label,
                    "MAE": values.get("mae", "n/a"),
                    "RMSE": values.get("rmse", "n/a"),
                    "R2": values.get("r2", "n/a"),
                    "MAPE (%)": values.get("mape", "n/a"),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No evaluation report is available. Run `python -m app.models.train` to generate one.")

    if metrics and metrics.get("xgboost_status"):
        st.caption(f"XGBoost: {metrics['xgboost_status']}")

    importance = getattr(model, "feature_importances_", None)
    if importance is not None:
        importance_frame = pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Importance": importance}
        ).sort_values("Importance", ascending=True)
        st.markdown("#### Random Forest feature importance")
        figure = px.bar(
            importance_frame,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[COLORS["blue"], COLORS["cyan"]],
        )
        st.plotly_chart(style_chart(figure, 380), width="stretch")
        st.caption("Importance describes how often a feature helps split the fitted trees; it is not a causal explanation.")


def main() -> None:
    st.set_page_config(page_title="Smart Demand Forecasting", page_icon="SD", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { --ink:#ECF4FF; --muted:#91A4BD; --blue:#3B82F6; --cyan:#36D6E7; }
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color:var(--ink); }
        [data-testid="stAppViewContainer"] { background-color:#08111F; background-image:linear-gradient(180deg,rgba(16,29,48,.46),rgba(8,17,31,.92)), repeating-linear-gradient(90deg,rgba(145,164,189,.025) 0,rgba(145,164,189,.025) 1px,transparent 1px,transparent 54px); }
        [data-testid="stHeader"] { background:rgba(8,17,31,.72); }
        [data-testid="stSidebar"] { background:#0C1727; border-right:1px solid rgba(145,164,189,.14); }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { font-size:14px; }
        h1, h2, h3 { letter-spacing:0 !important; color:#ECF4FF !important; }
        h1 { font-weight:800 !important; font-size:38px !important; }
        h2 { font-size:25px !important; }
        h3 { font-size:19px !important; }
        [data-testid="stMetric"] { background:#101D30; border:1px solid rgba(145,164,189,.15); border-top:2px solid #3B82F6; padding:16px 18px; border-radius:6px; min-height:112px; }
        [data-testid="stMetricLabel"] { color:#91A4BD !important; font-size:12px !important; }
        [data-testid="stMetricValue"] { color:#ECF4FF !important; font-size:25px !important; }
        [data-testid="stMetricDelta"] { font-family:'DM Mono',monospace; font-size:11px; }
        .hero { border-bottom:1px solid rgba(145,164,189,.17); padding:10px 0 25px; margin-bottom:22px; }
        .hero-kicker, .section-kicker { color:#36D6E7; font:500 11px 'DM Mono',monospace; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:8px; }
        .hero-copy { color:#91A4BD; font-size:14px; margin-top:-14px; }
        .brand { color:#ECF4FF; font-weight:800; font-size:18px; letter-spacing:0; }
        .brand small { color:#36D6E7; font:500 10px 'DM Mono',monospace; display:block; margin-top:5px; letter-spacing:1px; }
        div[data-testid="stDataFrame"] { border:1px solid rgba(145,164,189,.16); border-radius:6px; }
        div[data-testid="stDownloadButton"] button { border:1px solid rgba(54,214,231,.45); color:#ECF4FF; background:#102038; }
        div[data-testid="stDownloadButton"] button:hover { border-color:#36D6E7; color:#36D6E7; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="brand">SD / ENERGY<small>DEMAND INTELLIGENCE</small></div>', unsafe_allow_html=True)
    st.sidebar.markdown(" ")
    page = st.sidebar.radio("Workspace", ["Overview", "Consumption", "Forecast", "Model insights"], label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption("Hourly energy demand")

    try:
        history = load_history()
        model, model_label = load_model()
        metrics, metrics_label = load_metrics()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    st.markdown(
        f'<div class="hero"><div class="hero-kicker">ENERGY OPERATIONS / FORECASTING</div>'
        f'<h1>Smart Demand Forecasting</h1>'
        f'<div class="hero-copy">Explore {len(history):,} observed hours, understand demand patterns, and model the next hours ahead.</div></div>',
        unsafe_allow_html=True,
    )
    if model_label == "Preserved historical model":
        st.warning("The leakage-safe model has not been generated yet. This view is using the preserved historical artifact; run `python -m app.models.train` for corrected training.")

    if page == "Overview":
        render_overview(history, metrics)
    elif page == "Consumption":
        render_consumption(history)
    elif page == "Forecast":
        render_forecast(history, model)
    else:
        render_model_insights(model, metrics, model_label, metrics_label)


if __name__ == "__main__":
    main()