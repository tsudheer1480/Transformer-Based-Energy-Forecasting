import os
# limit numpy / torch threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hybrid_multitask import HybridMultiTask
from config import LOOKBACK, DEVICE
from visualization.future_forecast_plot import plot_future_forecast
from evaluation.evaluate_multiscale import evaluate_multiscale
from interface.forecast_interface import generate_forecast_outputs
from explainability.dynamic_feature_importance import compute_dynamic_feature_importance

torch.set_num_threads(1)
torch.set_grad_enabled(False)
device = torch.device(DEVICE)

print("Using device:", device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "results", "models",
                          "New_hybrid_model_final_80epochs.pth")

FEATURE_SCALER_PATH = os.path.join(BASE_DIR, "..", "results",
                                   "models", "feature_scaler.pkl")

TARGET_SCALER_PATH = os.path.join(BASE_DIR, "..", "results",
                                  "models", "target_scaler.pkl")

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

INPUT_DIM = 17

model = None
feature_scaler = None
target_scaler = None

def get_model_and_scalers():

    global model, feature_scaler, target_scaler

    if model is None:
        print("Lazy loading model...")

        model_local = HybridMultiTask(input_dim=INPUT_DIM)

        state_dict = torch.load(MODEL_PATH, map_location=device)

        # Handle positional encoding mismatch safely
        if "positional_encoding.pe" in state_dict:
            pretrained_pe = state_dict["positional_encoding.pe"]

            if pretrained_pe.shape[1] != LOOKBACK:
                print(f"Adjusting PE from {pretrained_pe.shape[1]} → {LOOKBACK}")

                if pretrained_pe.shape[1] > LOOKBACK:
                    # truncate
                    state_dict["positional_encoding.pe"] = pretrained_pe[:, :LOOKBACK, :]
                else:
                    # pad
                    pad_size = LOOKBACK - pretrained_pe.shape[1]
                    pad = torch.zeros((1, pad_size, pretrained_pe.shape[2]))
                    state_dict["positional_encoding.pe"] = torch.cat([pretrained_pe, pad], dim=1)

        model_local.load_state_dict(state_dict, strict=False)

        model_local.to(device)
        model_local.eval()

        fs = joblib.load(FEATURE_SCALER_PATH)
        ts = joblib.load(TARGET_SCALER_PATH)

        model = model_local
        feature_scaler = fs
        target_scaler = ts

        print("Model loaded successfully (lazy).")

    return model, feature_scaler, target_scaler
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def generate_features(df):

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    weather_cols = ["solar","wind","temperature","humidity","wind_speed","precipitation"]

    for col in weather_cols:
        if col not in df.columns:
            df[col] = 0

    df["load_lag_1"] = df["load"].shift(1)
    df["load_lag_24"] = df["load"].shift(24)
    df["load_lag_168"] = df["load"].shift(168)

    df["rolling_mean_24"] = df["load"].rolling(24).mean()
    df["rolling_std_24"] = df["load"].rolling(24).std()

    df = df.dropna().reset_index(drop=True)

    return df

@app.get("/")
def home():
    return {
        "message": "Welcome to the Energy Forecasting API. Use /run_model to get forecasts."
    }

@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/run_model")
async def run_model(
    file: UploadFile = File(...),
    mode: str = Form("forecast")
):

    try:

        df = pd.read_csv(file.file, low_memory=False)

        # convert numeric columns safely
        for col in df.columns:
            if col != "time":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # remove bad rows
        df = df.dropna().reset_index(drop=True)
        if len(df) > 15000:
            df = df.tail(15000)

        model, feature_scaler, target_scaler = get_model_and_scalers()

        # Minimum raw rows check
        if len(df) < 200:
            raise HTTPException(
                status_code=400,
                detail="Dataset too small. Upload at least 200 rows."
            )

        # Generate lag / rolling features

        # IMPORTANT: check again after feature engineering
        if len(df) <= LOOKBACK:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset insufficient after preprocessing. Need at least {LOOKBACK+200} rows."
            )
        # dataset size validation after feature generation
        if mode.lower() == "evaluate":
            if len(df) < 1600:
                raise HTTPException(
                    status_code=400,
                    detail="Dataset is not sufficient for evaluation. At least 1600 rows required."
                )
        df = generate_features(df)

        feature_cols = [
            'hour','day_of_week','month','is_weekend','is_holiday',
            'load','solar','wind','temperature','humidity',
            'wind_speed','precipitation',
            'load_lag_1','load_lag_24','load_lag_168',
            'rolling_mean_24','rolling_std_24'
        ]

        with torch.no_grad():
            forecast_results = generate_forecast_outputs(df, feature_cols, target_scaler, model)

        if forecast_results is None:
            raise HTTPException(
                status_code=500,
                detail="Forecast generation failed."
            )
        feature_importance = compute_dynamic_feature_importance(
            model,
            df.tail(LOOKBACK * 2),
            feature_cols,
            device,
            LOOKBACK
        )

        top_features = [
            {"feature": f[0], "score": round(f[1],4)}
            for f in feature_importance[:8]
        ]
        # top_features = []
        last_time = pd.to_datetime(df["time"].iloc[-1])

        path_24 = plot_future_forecast(
            start_time=last_time,
            predictions=[x["load_mw"] for x in forecast_results["forecast_24h"]],
            horizon_hours=24,
            title="24 Hour Load Forecast",
            horizon_label="24H",
            save_dir=STATIC_DIR
        )

        path_7d = plot_future_forecast(
            start_time=last_time,
            predictions=[x["load_mw"] for x in forecast_results["forecast_7d"]],
            horizon_hours=7,
            title="7 Day Load Forecast",
            horizon_label="7D",
            save_dir=STATIC_DIR
        )

        path_30d = plot_future_forecast(
            start_time=last_time,
            predictions=[x["load_mw"] for x in forecast_results["forecast_30d"]],
            horizon_hours=30,
            title="30 Day Load Forecast",
            horizon_label="30D",
            save_dir=STATIC_DIR
        )

        response = {
            "mode": mode,
            "last_available_time": forecast_results["last_available_time"],
            "24h_forecast": forecast_results["forecast_24h"],
            "7d_forecast": forecast_results["forecast_7d"],
            "30d_forecast": forecast_results["forecast_30d"],
            "explanations": forecast_results["explanations"],

            "feature_influence": top_features,

            "graphs": {
                "24h": f"/static/{os.path.basename(path_24)}",
                "7d": f"/static/{os.path.basename(path_7d)}",
                "30d": f"/static/{os.path.basename(path_30d)}"
            }
        }

        if mode.lower() == "evaluate":

            df_eval = df.copy()

            feature_cols_no_target = [c for c in feature_cols if c != "load"]

            df_eval[feature_cols_no_target] = feature_scaler.transform(
                df_eval[feature_cols_no_target]
            )

            df_eval["load"] = np.log1p(df_eval["load"])
            df_eval["load"] = target_scaler.transform(df_eval[["load"]])

            evaluation_results = evaluate_multiscale(
                df_eval,
                feature_cols,
                target_scaler
            )

            clean_results = {k: float(v) for k, v in evaluation_results.items()}

            response["evaluation"] = clean_results

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))