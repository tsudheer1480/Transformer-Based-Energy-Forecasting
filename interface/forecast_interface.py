import torch
import pandas as pd
import numpy as np
import datetime
import os
import plotly.graph_objects as go

from models.hybrid_multitask import HybridMultiTask
from config import DEVICE, MODEL_PATH, MODEL_CONFIG,LOOKBACK

# Explainability imports
from explainability.explain_dynamic import dynamic_summary
from explainability.explain_academic import academic_explanation
from explainability.explain_attention import attention_explanation
from explainability.explain_features import feature_explanation


def interactive_forecast(df, feature_cols, target_scaler):


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

    # ==============================
    # Load model
    # ==============================

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

    print("Select Forecast Type:")
    print("1 → Next 24 Hours")
    print("2 → Next 7 Days")
    print("3 → Next 30 Days")

    choice = input("Enter choice (1/2/3): ")

    os.makedirs("results/plots", exist_ok=True)

    # =========================================================
    # 24 HOURS
    # =========================================================

    if choice == "1":

        future_times = [
            last_date + datetime.timedelta(hours=i+1)
            for i in range(24)
        ]

        print("\n24-Hour Forecast:\n")
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

        # ===== Explainability =====
        print("\n===== Forecast Explanation =====\n")

        print(dynamic_summary(p50_24[:24], horizon="24H"))
        print("\n", academic_explanation(p50_24[:24], horizon="24H"))
        print("\n", attention_explanation(attention, horizon="24H"))
        print("\n", feature_explanation(feature_cols))

    # =========================================================
    # 7 DAYS
    # =========================================================

    elif choice == "2":

        daily_values = []
        dates = []

        print("\n7-Day Forecast:\n")

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

        # ===== Explainability =====
        print("\n===== Forecast Explanation =====\n")

        print(dynamic_summary(daily_values, horizon="7D"))
        print("\n", academic_explanation(daily_values, horizon="7D"))
        print("\n", attention_explanation(attention, horizon="7D"))
        print("\n", feature_explanation(feature_cols))

    # =========================================================
    # 30 DAYS
    # =========================================================

    elif choice == "3":

        daily_values = []
        dates = []

        print("\n30-Day Forecast:\n")

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

        # ===== Explainability =====
        print("\n===== Forecast Explanation =====\n")

        print(dynamic_summary(daily_values, horizon="30D"))
        print("\n", academic_explanation(daily_values, horizon="30D"))
        print("\n", attention_explanation(attention, horizon="30D"))
        print("\n", feature_explanation(feature_cols))

    else:
        print("Invalid selection.")