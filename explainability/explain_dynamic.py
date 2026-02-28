import numpy as np

def dynamic_summary(values, horizon=""):

    values = np.array(values)

    mean_val = np.mean(values)
    max_val = np.max(values)
    min_val = np.min(values)
    std_val = np.std(values)

    peak_index = np.argmax(values)
    low_index = np.argmin(values)

    trend = "increasing" if values[-1] > values[0] else "decreasing"

    summary = f"""
Trend Analysis ({horizon}):

• Average projected load: {mean_val:,.2f} MW
• Maximum projected load: {max_val:,.2f} MW
• Minimum projected load: {min_val:,.2f} MW
• Volatility (Std Dev): {std_val:,.2f} MW
• Overall trend direction: {trend}

The forecast indicates structural demand variation 
with peak occurring at index position {peak_index}
and minimum at position {low_index}.
"""

    return summary