# ⚡ Transformer-Based Energy Consumption Forecasting

## 📌 Project Overview
This project focuses on long-term energy consumption forecasting using advanced Transformer-based deep learning models. The objective is to accurately predict future electricity load using historical consumption data, weather features, and time-based variables.

The system compares traditional statistical models with modern Transformer architectures to improve forecasting accuracy, scalability, and interpretability.

---

## 🎯 Objectives

- Forecast multi-step future energy load
- Capture long-term dependencies in time-series data
- Compare baseline models with advanced Transformer models
- Improve long-horizon prediction accuracy
- Provide interpretable forecasting results

---

## 🧠 Models Implemented

### 🔹 Baseline Models
- ARIMA
- SARIMA
- LSTM
- DeepAR
- MQRNN

### 🔹 Transformer-Based Models
- Temporal Fusion Transformer (TFT)
- Informer

---

## 📊 Dataset

The dataset includes:

- Electricity Load
- Solar Generation
- Wind Generation
- Wind Onshore / Offshore
- Hour
- Day of Week
- Month
- Holiday Indicators
- Lag Features

Data is preprocessed with normalization and lag-based feature engineering.

---

## 📁 Project Structure

```
energy_forecasting/
│
├── config.py                   # Global configuration settings
├── main.py                     # Main execution pipeline
│
├── data/
│   └── processed/
│       └── final_energy_forecasting_dataset.csv
│
├── evaluation/
│   └── evaluate_multiscale.py  # MAE, RMSE, coverage metrics
│
├── explainability/
│   ├── explain_academic.py
│   ├── explain_attention_academic.py
│   └── explain_features.py
│
├── interface/
│   └── forecast_interface.py   # Interactive forecast system
│
├── models/
│   └── hybrid_multitask.py     # Transformer-based forecasting model
│
├── preprocessing/
│   └── dataset_multitask.py    # Dataset preparation & sliding window
│
├── production/
│   └── daily_forecast.py       # Real-time deployment script
│
├── results/
│   ├── models/                 # Saved model weights
│   ├── plots/                  # Forecast visualizations
│   └── reports/                # Generated summaries
│
└── training/
    └── train.py                # Model training (50 epochs)


```

---

## 📈 Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MSE (Mean Squared Error)
- Quantile Loss (for probabilistic forecasting)

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Statsmodels

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```
git clone https://github.com/yourusername/energy-forecasting.git
cd energy-forecasting
```

### 2️⃣ Create Virtual Environment
```
python -m venv energy_env
energy_env\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Train the Model
```
python main.py
```

### 5️⃣ Evaluate the Model
```
python evaluate.py
```

---

## 📊 Results

Transformer-based models (TFT and Informer) outperform traditional LSTM and ARIMA models in:

- Long-horizon forecasting
- Capturing seasonal patterns
- Handling multiple input features
- Providing interpretable outputs

---

## 🔍 Key Contributions

- Implementation of Temporal Fusion Transformer (TFT)
- Implementation of Informer for long-sequence forecasting
- Advanced feature engineering with lag variables
- Multi-horizon probabilistic forecasting
- Comparative analysis of classical vs deep learning models

---

## 👨‍💻 Author

Sudheer Tantapureddy  
B.Tech – SRKR Engineering College  
Data Science & Machine Learning Enthusiast  

---

## 📜 License

This project is developed for academic and research purposes.
