import numpy as np

def generate_research_summary_24h(p50_values):

    trend = p50_values[-1] - p50_values[0]
    peak_idx = np.argmax(p50_values)
    peak_value = p50_values[peak_idx]
    avg_load = np.mean(p50_values)
    std_dev = np.std(p50_values)

    if trend > 0:
        trend_text = "an overall increasing demand trajectory"
    elif trend < 0:
        trend_text = "a gradual declining demand trajectory"
    else:
        trend_text = "a relatively stable demand pattern"

    if 17 <= peak_idx <= 22:
        peak_text = "evening peak behavior consistent with typical consumption cycles"
    else:
        peak_text = "non-conventional peak timing relative to standard daily cycles"

    variability = std_dev / avg_load

    if variability < 0.05:
        stability_text = "low variability indicating a stable and predictable load profile"
    elif variability < 0.10:
        stability_text = "moderate variability suggesting manageable fluctuations"
    else:
        stability_text = "higher variability reflecting dynamic demand conditions"

    summary = f"""
    The 24-hour ahead load forecast exhibits {trend_text}, 
    with peak demand occurring at hour {peak_idx + 1} reaching approximately {peak_value:,.0f} MW. 
    The average projected load is estimated at {avg_load:,.0f} MW. 

    The intraday demand distribution demonstrates {peak_text}. 
    Statistical dispersion analysis indicates {stability_text}. 
    Overall, the forecast suggests strong short-term dependency on recent consumption patterns 
    while maintaining structural consistency with typical daily electricity demand cycles.
    """

    return summary


def generate_research_summary_multi(daily_values, horizon_label):

    trend = daily_values[-1] - daily_values[0]
    peak_day = np.argmax(daily_values) + 1
    peak_value = np.max(daily_values)
    avg_load = np.mean(daily_values)
    std_dev = np.std(daily_values)

    if trend > 0:
        trend_text = "a progressive upward demand trend"
    elif trend < 0:
        trend_text = "a gradual decline in demand levels"
    else:
        trend_text = "a relatively stable demand structure"

    variability = std_dev / avg_load

    if variability < 0.05:
        stability_text = "minimal inter-day variation"
    elif variability < 0.10:
        stability_text = "moderate inter-day variability"
    else:
        stability_text = "considerable inter-day fluctuations"

    summary = f"""
    The {horizon_label} forecast reveals {trend_text}, 
    with the highest projected demand observed on day {peak_day} 
    reaching approximately {peak_value:,.0f} MW. 
    The mean forecasted load over the period is {avg_load:,.0f} MW.

    Variability analysis indicates {stability_text}, 
    suggesting that the model captures medium-term structural demand dynamics 
    while accounting for temporal dependencies embedded within historical load patterns.
    """

    return summary
