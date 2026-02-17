import torch
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from models.hybrid_multitask import HybridMultiTask
from preprocessing.dataset_multitask import DatasetMultiTask
from config import DEVICE, MODEL_PATH
from explain_academic import (
    generate_research_summary_24h,
    generate_research_summary_multi
)

sns.set_style("darkgrid")
sns.set_context("talk")


def interactive_forecast(df, feature_cols, target_scaler):

    last_date = pd.to_datetime(df["time"].iloc[-1])

    print("\n==========================================")
    print(f"Last Available Data Date : {last_date.strftime('%Y-%m-%d %A %H:%M')}")
    print("Forecasts start from next hour onward.")
    print("==========================================\n")

    dataset = DatasetMultiTask(df, feature_cols)
    x, _, _, _ = dataset[0]
    x = x.unsqueeze(0).to(DEVICE)

    model = HybridMultiTask(len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred24, pred7d, pred30d, _ = model(x)

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

    # =======================
    # 24 HOURS
    # =======================

    if choice == "1":

        forecast_times = [
            last_date + datetime.timedelta(hours=i+1)
            for i in range(24)
        ]

        print("\n24-Hour Forecast:\n")

        for i in range(24):
            print(f"{forecast_times[i].strftime('%Y-%m-%d %A %H:%M')} → {p50_24[i]:,.2f} MW")

        plt.figure(figsize=(12, 6))

        sns.lineplot(x=forecast_times, y=p50_24, linewidth=2.5, color="royalblue")

        peak_idx = np.argmax(p50_24)
        plt.scatter(forecast_times[peak_idx], p50_24[peak_idx], color="red", s=120)
        plt.title("24-Hour Load Forecast", fontweight="bold")
        plt.xlabel("Date & Time")
        plt.ylabel("Load (MW)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\nResearch-Level Summary:\n")
        print(generate_research_summary_24h(p50_24))

    # =======================
    # 7 DAYS
    # =======================

    elif choice == "2":

        daily_values = []
        dates = []

        print("\n7-Day Forecast:\n")

        for d in range(7):
            target_date = last_date + datetime.timedelta(days=d+1)
            daily_avg = p50_7d[d*24:(d+1)*24].mean()

            daily_values.append(daily_avg)
            dates.append(target_date.strftime("%Y-%m-%d"))

            print(f"{target_date.strftime('%Y-%m-%d %A')} → {daily_avg:,.2f} MW")

        plt.figure(figsize=(10, 5))
        sns.lineplot(x=dates, y=daily_values, marker="o", linewidth=2.5, color="seagreen")
        plt.title("7-Day Average Load Forecast", fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Average Load (MW)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\nResearch-Level Summary:\n")
        print(generate_research_summary_multi(daily_values, "7-day"))

    # =======================
    # 30 DAYS
    # =======================

    elif choice == "3":

        daily_values = []
        dates = []

        print("\n30-Day Forecast:\n")

        for d in range(30):
            target_date = last_date + datetime.timedelta(days=d+1)
            daily_avg = p50_30d[d*24:(d+1)*24].mean()

            daily_values.append(daily_avg)
            dates.append(target_date.strftime("%m-%d"))

            print(f"{target_date.strftime('%Y-%m-%d %A')} → {daily_avg:,.2f} MW")

        plt.figure(figsize=(12, 5))
        sns.lineplot(x=dates, y=daily_values, linewidth=2, color="purple")
        plt.title("30-Day Average Load Forecast", fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Average Load (MW)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\nResearch-Level Summary:\n")
        print(generate_research_summary_multi(daily_values, "30-day"))

    else:
        print("Invalid selection. Please choose 1, 2, or 3.")
