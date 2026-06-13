# -*- coding: utf-8 -*-
"""
SVR_Regressor
This script trains, saves, and evaluates Support Vector Regression (SVR) models 
for Remaining Useful Life (RUL) prediction across various denoised signals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter
import joblib

# =====================================================================
# STEP 1: Configuration & Data Loading
# =====================================================================
# Update this path to where your consolidated dataset is located
file_path = r"[INSERT_DATASET_CSV_PATH_HERE]"
df = pd.read_csv(file_path)

# List of signal processing techniques to evaluate
signal_columns = [
    "noisy_signal", "SMA", "MF", "KF", "EWMA", "FFT", "Wavelet", 
    "ARIMA", "Kalman", "Particle", "Linear_GAN", "Linear_Autoencoder",
    "Conv_GAN", "Conv_Autoencoder", "LSTM_GAN", "LSTM_Autoencoder", "EEMD"
]

window_size = 100
stride = 20
time_per_point = 1 / 10000  # Sample time in seconds

# Define output directories (Update these placeholders)
model_save_dir = r"[INSERT_MODEL_SAVE_DIRECTORY_HERE]"
chart_save_dir = r"[INSERT_CHART_SAVE_DIRECTORY_HERE]"

# Ensure output directories exist
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(chart_save_dir, exist_ok=True)

# =====================================================================
# STEP 2: Data Preparation Function
# =====================================================================
def prepare_data(df, signal_col):
    """
    Slices the continuous time-series data into overlapping windows.
    Returns the feature windows, corresponding RUL labels, and timestamps.
    """
    windows, labels, time_stamps = [], [], []
    for i in range(0, len(df) - window_size, stride):
        window = df.iloc[i:i+window_size]
        windows.append(window[[signal_col, 'RUL']].values)
        labels.append(window['RUL'].iloc[-1])
        
        # Calculate the timestamp for the end of the window
        end_time = (i + window_size - 1) * time_per_point
        time_stamps.append(end_time)
        
    return np.array(windows), np.array(labels), np.array(time_stamps)

# =====================================================================
# STEP 3: Training and Evaluation Pipeline
# =====================================================================
for col in signal_columns:
    print(f"\n🔍 Evaluating model for signal: {col}")
    
    # Check for and handle NaN values by replacing them with 0
    if df[col].isnull().sum() > 0:
        print(f"⚠️ Column {col} has {df[col].isnull().sum()} NaN values. Replacing with 0.")
        df[col].fillna(0, inplace=True)
        
    # --- Data Preparation ---
    X, y, time_stamps = prepare_data(df, col)
    
    # Flatten data from (samples, window_size, 1) -> (samples, window_size)
    X_flat = X.reshape((X.shape[0], -1))

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
        X_flat, y, time_stamps, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # --- Model Training ---
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
    svr_model.fit(X_train_scaled, y_train)

    # Save the trained SVR model
    model_save_path = os.path.join(model_save_dir, f"RUL_model_SVR_{col}.pkl")
    joblib.dump(svr_model, model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    # --- Model Evaluation ---
    predictions = svr_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"📊 Metrics -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # --- Plotting Results ---
    # Note: Sorting by time_test so the plot lines draw correctly left-to-right
    sorted_indices = np.argsort(time_test)
    sorted_time = time_test[sorted_indices]
    sorted_y_test = y_test[sorted_indices]
    sorted_preds = predictions[sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_time, sorted_y_test, label="Actual RUL", color='blue')
    
    # Apply smoothing to predictions
    smoothed_preds = savgol_filter(sorted_preds, 51, 3)
    plt.plot(sorted_time, smoothed_preds, label="Smoothed Prediction", color='red')
    
    plt.scatter(sorted_time, sorted_preds, s=10, color='orange', label="Raw Prediction")
    plt.title(f"RUL Prediction vs Actual RUL - SVR ({col})")
    plt.xlabel("Time (s)")
    plt.ylabel("RUL")
    plt.legend()

    # Save Chart
    chart_save_path = os.path.join(chart_save_dir, f"RUL_model_SVR_{col}.png")
    plt.savefig(chart_save_path)
    plt.close()
    print(f"📈 Chart saved to: {chart_save_path}")