import numpy as np

def dynamic_summary(values, horizon, timestamps=None):

    values = np.array(values)

    avg = np.mean(values)
    max_val = np.max(values)
    min_val = np.min(values)
    std_val = np.std(values)

    peak_index = int(np.argmax(values))
    min_index = int(np.argmin(values))

    trend = "increasing" if values[-1] > values[0] else "decreasing"

    # optional windows
    peak_start = max(0, peak_index - 1)
    peak_end = min(len(values)-1, peak_index + 1)

    min_start = max(0, min_index - 1)
    min_end = min(len(values)-1, min_index + 1)

    if timestamps is not None:

        peak_time = timestamps[peak_index]
        min_time = timestamps[min_index]

        peak_start_t = timestamps[peak_start]
        peak_end_t = timestamps[peak_end]

        min_start_t = timestamps[min_start]
        min_end_t = timestamps[min_end]

        # ---------- 24H ----------
        if horizon == "24H":

            peak_text = f"{peak_time.strftime('%H:%M')} hours"
            min_text = f"{min_time.strftime('%H:%M')} hours"

            peak_window = f"{peak_start_t.strftime('%H:%M')} – {peak_end_t.strftime('%H:%M')}"
            min_window = f"{min_start_t.strftime('%H:%M')} – {min_end_t.strftime('%H:%M')}"

        elif horizon == "7D":

            peak_text = f"{peak_time.strftime('%Y-%m-%d')}"
            min_text = f"{min_time.strftime('%Y-%m-%d')}"

            peak_window = "Evening hours (17:00 – 21:00)"
            min_window = "Early morning hours (02:00 – 06:00)"


        else:  # 30D

            peak_text = f"{peak_time.strftime('%Y-%m-%d')}"
            min_text = f"{min_time.strftime('%Y-%m-%d')}"

            peak_window = "Evening hours (17:00 – 21:00)"
            min_window = "Early morning hours (02:00 – 06:00)"

    else:

        peak_text = f"index {peak_index}"
        min_text = f"index {min_index}"

        peak_window = f"{peak_start} – {peak_end}"
        min_window = f"{min_start} – {min_end}"

    text = f"""
Trend Analysis ({horizon}):

• Average projected load: {avg:.2f} MW
• Maximum projected load: {max_val:.2f} MW
• Minimum projected load: {min_val:.2f} MW
• Volatility (Std Dev): {std_val:.2f} MW
• Overall trend direction: {trend}

Peak demand occurs at {peak_text}.
Nearby peak window: {peak_window}.

Minimum demand occurs at {min_text}.
Nearby minimum window: {min_window}.
"""

    return text