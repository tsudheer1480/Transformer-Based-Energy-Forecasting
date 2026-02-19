import pandas as pd
import numpy as np
from config import FEATURE_CONFIG, TARGET_COL
def fixed_model_time_window_validation(df, feature_cols, target_scaler):

    print("\n===== Fixed Model Time-Window Validation (24H Only) =====")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    from evaluation.evaluate_multiscale import evaluate_multiscale

    test_days = 90  # Larger window for stability
    LOOKBACK = 168
    HORIZON_24 = 24
    MIN_REQUIRED_ROWS = LOOKBACK + HORIZON_24

    start_date = df["time"].min()
    end_date = df["time"].max()

    current_start = start_date

    all_mae = []

    while True:

        window_end = current_start + pd.Timedelta(days=test_days)

        if window_end > end_date:
            break

        test_window = df[
            (df["time"] >= current_start) &
            (df["time"] < window_end)
        ]

        if len(test_window) < MIN_REQUIRED_ROWS:
            print("Window skipped (insufficient rows).")
        else:
            print(f"Evaluating: {current_start.date()} → {window_end.date()}")

            metrics = evaluate_multiscale(test_window, feature_cols)

            if metrics["24H_MAE"] is not None:
                all_mae.append(metrics["24H_MAE"])

        current_start += pd.Timedelta(days=test_days)

    print("\n===== Stability Summary =====")

    if len(all_mae) > 0:
        print(f"Average 24H MAE: {np.mean(all_mae):.4f}")
        print(f"MAE Std Dev: {np.std(all_mae):.4f}")
    else:
        print("No valid windows for evaluation.")

    print("\nFixed model validation complete.")
