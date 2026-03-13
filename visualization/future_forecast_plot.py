import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np


def plot_future_forecast(start_time,
                         predictions,
                         horizon_hours,
                         title,
                         horizon_label,
                         save_dir):

    predictions = np.array(predictions)

    # -----------------------
    # Generate timestamps
    # -----------------------

    if horizon_label == "24H":

        x_values = pd.date_range(
            start=start_time + pd.Timedelta(hours=1),
            periods=len(predictions),
            freq="h"
        )

        xaxis_config = dict(
            title="Time",
            showgrid=True,
            tickformat="%H:%M",
            tickformatstops=[
                dict(dtickrange=[None, 3600000], value="%H:%M"),
                dict(dtickrange=[3600000, None], value="%b %d\n%H:%M")
            ]
        )

    elif horizon_label == "7D":

        x_values = pd.date_range(
            start=start_time + pd.Timedelta(days=1),
            periods=len(predictions),
            freq="D"
        )

        xaxis_config = dict(
            title="Date",
            showgrid=True,
            tickformat="%b %d"
        )

    elif horizon_label == "30D":

        x_values = pd.date_range(
            start=start_time + pd.Timedelta(days=1),
            periods=len(predictions),
            freq="D"
        )

        xaxis_config = dict(
            title="Date",
            showgrid=True,
            tickformat="%b %d\n%Y"
        )

    else:

        x_values = range(len(predictions))

        xaxis_config = dict(
            title="Index",
            showgrid=True
        )

    y_values = predictions

    # -----------------------
    # Safety check
    # -----------------------

    if len(y_values) == 0:

        fig = go.Figure()
        fig.update_layout(title=title)

        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f"forecast_{horizon_label}.html")
        fig.write_html(file_path)

        return file_path


    # -----------------------
    # Peak & Minimum
    # -----------------------

    peak_index = int(np.argmax(y_values))
    min_index = int(np.argmin(y_values))


    # -----------------------
    # Graph
    # -----------------------

    fig = go.Figure()

    fig.add_trace(go.Scatter(

        x=x_values,
        y=y_values,

        mode="lines+markers",

        line=dict(
            width=4,
            shape="spline",
            smoothing=1.2
        ),

        name="Forecast Load",

        hovertemplate="<b>%{x}</b><br>Load: %{y:.2f} MW<extra></extra>"
    ))


    # Peak marker
    fig.add_trace(go.Scatter(

        x=[x_values[peak_index]],
        y=[y_values[peak_index]],

        mode="markers",

        marker=dict(
            size=14,
            color="red"
        ),

        name="Peak Load"
    ))


    # Minimum marker
    fig.add_trace(go.Scatter(

        x=[x_values[min_index]],
        y=[y_values[min_index]],

        mode="markers",

        marker=dict(
            size=14,
            color="green"
        ),

        name="Minimum Load"
    ))


    # -----------------------
    # Layout
    # -----------------------

    fig.update_layout(

        title=title,

        xaxis=xaxis_config,

        yaxis=dict(
            title="Load (MW)",
            showgrid=True
        ),

        template="plotly_white",

        hovermode="x unified"

    )


    # -----------------------
    # Save graph
    # -----------------------

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f"forecast_{horizon_label}.html")

    fig.write_html(file_path)

    return file_path