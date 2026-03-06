import plotly.graph_objects as go
import pandas as pd
import os


def plot_future_forecast(start_time,
                         predictions,
                         horizon_hours,
                         title,
                         horizon_label,
                         save_dir):

    """
    Generates and saves future forecast graph as HTML.

    Parameters:
    -----------
    start_time : datetime
    predictions : array-like
    horizon_hours : int
    title : str
    horizon_label : str  (e.g., "24H", "7D", "30D")
    save_dir : str  (static folder path)

    Returns:
    --------
    file_path : str
    """

    # Generate future timestamps
    future_dates = pd.date_range(
        start=start_time,
        periods=horizon_hours,
        freq="h"
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name='Forecasted Load',
        hovertemplate='Time: %{x}<br>Load: %{y:.2f} MW'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Future Time",
        yaxis_title="Load (MW)",
        template="plotly_white",
        hovermode="x unified"
    )

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    file_name = f"forecast_{horizon_label}.html"
    file_path = os.path.join(save_dir, file_name)

    fig.write_html(file_path)

    return file_path