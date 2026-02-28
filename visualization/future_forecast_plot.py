import plotly.graph_objects as go
import pandas as pd

def plot_future_forecast(start_time, predictions, horizon_hours, title):

    future_dates = pd.date_range(
        start=start_time,
        periods=horizon_hours,
        freq="H"
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

    return fig