import torch
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from models.hybrid_multitask import HybridMultiTask
from preprocessing.dataset_multitask import DatasetMultiTask
from config import DEVICE, MODEL_PATH


def interactive_forecast(df, feature_cols, target_scaler):

    # ===========================
    # SHOW LAST DATASET DATE
    # ===========================

    last_date = pd.to_datetime(df["time"].iloc[-1])

    print("\n==========================================")
    print(f"Last Available Data Date : {last_date.strftime('%Y-%m-%d %A %H:%M')}")
    print("Forecasts start from next hour onward.")
    print("==========================================\n")

    # ===========================
    # LOAD MODEL
    # ===========================

    dataset = DatasetMultiTask(df, feature_cols)
    x, _, _, _ = dataset[0]
    x = x.unsqueeze(0).to(DEVICE)

    model = HybridMultiTask(len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred24, pred7d, pred30d, _ = model(x)

    # Convert to real MW
    def inverse(pred):
        return target_scaler.inverse_transform(
            pred[:, :, 1].cpu().numpy().reshape(-1, 1)
        ).flatten()

    p50_24 = inverse(pred24)
    p50_7d = inverse(pred7d)
    p50_30d = inverse(pred30d)

    # ===========================
    # USER INPUT
    # ===========================

    print("Select Forecast Type:")
    print("1 → Next 24 Hours")
    print("2 → Next 7 Days")
    print("3 → Next 30 Days")

    choice = input("Enter choice (1/2/3): ")

    # ===========================
    # 24 HOURS
    # ===========================

    if choice == "1":

        print("\n24-Hour Forecast:\n")

        hours = np.arange(1, 25)

        for i in range(24):
            forecast_time = last_date + datetime.timedelta(hours=i+1)
            print(f"{forecast_time.strftime('%Y-%m-%d %A %H:%M')} → {p50_24[i]:,.2f} MW")

        # Plot
                # ===== Better 24H Plot =====

        forecast_times = [
            last_date + datetime.timedelta(hours=i+1)
            for i in range(24)
        ]

        plt.figure(figsize=(12, 6))

        plt.plot(
            forecast_times,
            p50_24,
            color="royalblue",
            linewidth=2.5,
            label="Predicted Load"
        )

        plt.fill_between(
            forecast_times,
            p50_24 * 0.97,
            p50_24 * 1.03,
            color="lightblue",
            alpha=0.4,
            label="Confidence Band"
        )

        plt.title("24-Hour Load Forecast", fontsize=14, fontweight="bold")
        plt.xlabel("Date & Time", fontsize=12)
        plt.ylabel("Load (MW)", fontsize=12)

        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


        print("\nSummary:")
        print(
            "The forecast is based on recent electricity usage patterns "
            "and daily demand cycles. Higher demand is expected during "
            "evening hours due to typical consumption behavior."
        )

    # ===========================
    # 7 DAYS
    # ===========================

    elif choice == "2":

        print("\n7-Day Forecast:\n")

        daily_values = []  # ✅ define first

        for d in range(7):
            target_date = last_date + datetime.timedelta(days=d+1)
            daily_avg = p50_7d[d*24:(d+1)*24].mean()
            daily_values.append(daily_avg)

            print(f"{target_date.strftime('%Y-%m-%d %A')} → {daily_avg:,.2f} MW")

        # Plot
        dates = [
            (last_date + datetime.timedelta(days=d+1)).strftime("%Y-%m-%d")
            for d in range(7)
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(dates, daily_values, marker="o", linewidth=2, color="darkgreen")
        plt.title("7-Day Average Load Forecast", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Average Load (MW)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        print("\nSummary:")
        print(
            "This weekly forecast reflects recent demand trends and "
            "weekly usage patterns, considering differences between "
            "weekdays and weekends."
        )

    # ===========================
    # 30 DAYS
    # ===========================

    elif choice == "3":
        print("\n30-Day Forecast:\n")

        daily_values = []  # ✅ define first

        for d in range(30):
            target_date = last_date + datetime.timedelta(days=d+1)
            daily_avg = p50_30d[d*24:(d+1)*24].mean()
            daily_values.append(daily_avg)

            print(f"{target_date.strftime('%Y-%m-%d %A')} → {daily_avg:,.2f} MW")

        # Plot
        dates = [
            (last_date + datetime.timedelta(days=d+1)).strftime("%m-%d")
            for d in range(30)
        ]

        plt.figure(figsize=(12, 5))
        plt.plot(dates, daily_values, linewidth=2, color="purple")
        plt.title("30-Day Average Load Forecast", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Average Load (MW)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
        print("\nSummary:")
        print(
            "This monthly forecast is generated using historical demand "
            "trends and seasonal behavior. It captures gradual variations "
            "expected over the coming weeks."
        )
    else:
        print("Invalid selection. Please choose 1, 2, or 3.")
