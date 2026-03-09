import torch
import pandas as pd
import numpy as np
import datetime
import os
import plotly.graph_objects as go

from models.hybrid_multitask import HybridMultiTask
from config import DEVICE, MODEL_PATH, MODEL_CONFIG, LOOKBACK

# Explainability imports
from explainability.explain_dynamic import dynamic_summary
from explainability.explain_academic import academic_explanation
from explainability.explain_attention import attention_explanation
from explainability.explain_features import feature_explanation


def generate_forecast_outputs(df, feature_cols, target_scaler):
    if len(df) < LOOKBACK:
        print("Not enough data for forecasting.")
        return

    print("\n===== FUTURE FORECAST SYSTEM =====")

    last_date = pd.to_datetime(df["time"].iloc[-1])

    print("\n==========================================")
    print(f"Last Available Data Date : {last_date.strftime('%Y-%m-%d %A %H:%M')}")
    print("Forecasts start from next hour onward.")
    print("==========================================\n")

    # ==============================
    # Prepare input window
    # ==============================

    recent_window = df.iloc[-LOOKBACK:][feature_cols].values
    x = torch.tensor(recent_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # # ==============================
    # # Load model
    # # ==============================

    model = HybridMultiTask(len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred24, pred7d, pred30d, attention = model(x)

    def inverse(pred):
        return target_scaler.inverse_transform(
            pred[:, :, 1].cpu().numpy().reshape(-1, 1)
        ).flatten()

    p50_24 = inverse(pred24)
    p50_7d = inverse(pred7d)
    p50_30d = inverse(pred30d)

    os.makedirs("results/plots", exist_ok=True)

    # =========================================================
    # 24 HOURS FORECAST
    # =========================================================

    print("\n==============================")
    print("NEXT 24 HOURS FORECAST")
    print("==============================\n")

    future_times = [
        last_date + datetime.timedelta(hours=i+1)
        for i in range(24)
    ]

    for i in range(24):
        print(f"{future_times[i]} → {p50_24[i]:,.2f} MW")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_times,
        y=p50_24[:24],
        mode='lines+markers',
        hovertemplate='Time: %{x}<br>Load: %{y:.2f} MW'
    ))

    fig.update_layout(
        title="Next 24 Hours Forecast",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html("results/plots/forecast_24h.html")
    fig.show(renderer="browser")

    print("\n===== 24H Forecast Explanation =====\n")
    print(dynamic_summary(p50_24[:24], horizon="24H"))
    print("\n", academic_explanation(p50_24[:24], horizon="24H"))
    print("\n", attention_explanation(attention, horizon="24H"))
    print("\n", feature_explanation(feature_cols))


    # =========================================================
    # 7 DAYS FORECAST
    # =========================================================

    print("\n==============================")
    print("NEXT 7 DAYS FORECAST")
    print("==============================\n")

    daily_values = []
    dates = []

    for d in range(7):
        date = last_date + datetime.timedelta(days=d+1)
        avg = p50_7d[d*24:(d+1)*24].mean()

        daily_values.append(avg)
        dates.append(date)

        print(f"{date.strftime('%Y-%m-%d %A')} → {avg:,.2f} MW")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=daily_values,
        mode='lines+markers',
        hovertemplate='Date: %{x}<br>Load: %{y:.2f} MW'
    ))

    fig.update_layout(
        title="Next 7 Days Forecast",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html("results/plots/forecast_7days.html")
    fig.show(renderer="browser")

    print("\n===== 7D Forecast Explanation =====\n")
    print(dynamic_summary(daily_values, horizon="7D"))
    print("\n", academic_explanation(daily_values, horizon="7D"))
    print("\n", attention_explanation(attention, horizon="7D"))
    print("\n", feature_explanation(feature_cols))


    # =========================================================
    # 30 DAYS FORECAST
    # =========================================================

    print("\n==============================")
    print("NEXT 30 DAYS FORECAST")
    print("==============================\n")

    daily_values = []
    dates = []

    for d in range(30):
        date = last_date + datetime.timedelta(days=d+1)
        avg = p50_30d[d*24:(d+1)*24].mean()

        daily_values.append(avg)
        dates.append(date)

        print(f"{date.strftime('%Y-%m-%d %A')} → {avg:,.2f} MW")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=daily_values,
        mode='lines+markers',
        hovertemplate='Date: %{x}<br>Load: %{y:.2f} MW'
    ))

    fig.update_layout(
        title="Next 30 Days Forecast",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html("results/plots/forecast_30days.html")
    fig.show(renderer="browser")

    print("\n===== 30D Forecast Explanation =====\n")
    print(dynamic_summary(daily_values, horizon="30D"))
    print("\n", academic_explanation(daily_values, horizon="30D"))
    print("\n", attention_explanation(attention, horizon="30D"))
    print("\n", feature_explanation(feature_cols))


def generate_forecast_outputs(df, feature_cols, target_scaler, model):

    if len(df) < LOOKBACK:
        return None

    last_date = pd.to_datetime(df["time"].iloc[-1])

    recent_window = df.iloc[-LOOKBACK:][feature_cols].values
    x = torch.tensor(recent_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred24, pred7d, pred30d, attention = model(x)

    def inverse(pred):
        return target_scaler.inverse_transform(
            pred[:, :, 1].cpu().numpy().reshape(-1, 1)
        ).flatten()

    p50_24 = inverse(pred24)
    p50_7d = inverse(pred7d)
    p50_30d = inverse(pred30d)

    # ==============================
    # 24 HOURS FORECAST
    # ==============================

    future_times = [
        last_date + datetime.timedelta(hours=i+1)
        for i in range(24)
    ]

    forecast_24 = []

    for i in range(24):
        forecast_24.append({
            "time": future_times[i].strftime("%Y-%m-%d %H:%M"),
            "load_mw": float(p50_24[i])
        })

    # ==============================
    # 7 DAYS FORECAST
    # ==============================

    forecast_7 = []
    dates_7 = []

    for d in range(7):
        date = last_date + datetime.timedelta(days=d+1)
        avg = p50_7d[d*24:(d+1)*24].mean()

        forecast_7.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": date.strftime("%A"),
            "load_mw": float(avg)
        })

        dates_7.append(date)
    
    # ==============================
    # 30 DAYS FORECAST
    # ==============================

    forecast_30 = []
    dates_30 = []

    for d in range(30):
        date = last_date + datetime.timedelta(days=d+1)
        avg = p50_30d[d*24:(d+1)*24].mean()

        forecast_30.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": date.strftime("%A"),
            "load_mw": float(avg)
        })

        dates_30.append(date)

    # ==============================
    # Convert hourly predictions to daily values
    # ==============================

    daily_7 = []
    for d in range(7):
        avg = p50_7d[d*24:(d+1)*24].mean()
        daily_7.append(avg)

    daily_30 = []
    for d in range(30):
        avg = p50_30d[d*24:(d+1)*24].mean()
        daily_30.append(avg)

    # ==============================
    # Explanations
    # ==============================

    explanations = {

        "24H": {
            "trend": dynamic_summary(p50_24[:24], "24H", future_times),
            "academic": academic_explanation(p50_24[:24], "24H"),
            "attention": attention_explanation(attention, "24H"),
            "features": feature_explanation(feature_cols)
        },

        "7D": {
            "trend": dynamic_summary(daily_7, "7D", dates_7),
            "academic": academic_explanation(daily_7, "7D"),
            "attention": attention_explanation(attention, "7D"),
            "features": feature_explanation(feature_cols)
        },

        "30D": {
            "trend": dynamic_summary(daily_30, "30D", dates_30),
            "academic": academic_explanation(daily_30, "30D"),
            "attention": attention_explanation(attention, "30D"),
            "features": feature_explanation(feature_cols)
        }
    }

    return {
        "last_available_time": last_date.strftime("%Y-%m-%d %A %H:%M"),
        "forecast_24h": forecast_24,
        "forecast_7d": forecast_7,
        "forecast_30d": forecast_30,
        "explanations": explanations
    }