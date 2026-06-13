# -*- coding: utf-8 -*-
"""
RUL_LSTM_with_attention_optimize_by_optuna
This script builds, optimizes (via Optuna), and evaluates an LSTM + Self-Attention 
model for Remaining Useful Life (RUL) prediction.
"""

# =====================================================================
# STEP 1: Environment Setup (Google Colab specific)
# =====================================================================
# Uncomment these lines if running in a Google Colab notebook
# !pip install tensorflow scikit-learn matplotlib pandas numpy optuna
# from google.colab import drive
# drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter
import optuna

# =====================================================================
# STEP 2: Global Configuration & Data Loading
# =====================================================================
# Update this path to where your denoised CSV dataset is located
file_path = r"[INSERT_YOUR_DATASET_PATH_HERE]"
df = pd.read_csv(file_path)

# Configuration for sliding window approach
signal_columns = ["Denoised Signal"]
window_size = 1024  # Number of time steps per sequence
stride = 512        # Overlap step size
time_per_point = 1 / 10000  # Sample time in seconds

def prepare_data(df, signal_col):
    """
    Slices the continuous time-series data into windows (sequences) for LSTM input.
    Extracts the corresponding RUL label at the end of each window.
    """
    windows, labels = [], []
    for i in range(0, len(df) - window_size, stride):
        window = df.iloc[i:i+window_size]
        windows.append(window[[signal_col, 'RUL']].values)
        labels.append(window['RUL'].iloc[-1])
    return np.array(windows), np.array(labels)

# Generate features (X) and labels (y)
X, y = prepare_data(df, signal_columns[0])

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# STEP 3: Data Normalization
# =====================================================================
# Flatten the 3D data to 2D for scaling, then reshape back to 3D
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# =====================================================================
# STEP 4: Model Architecture Components
# =====================================================================
def self_attention_block(x, num_heads, key_dim=32):
    """
    Creates a Multi-Head Self-Attention block with residual connection 
    and layer normalization to capture long-range dependencies.
    """
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attention_out = layers.Add()([x, attention]) # Residual connection
    attention_out = layers.LayerNormalization()(attention_out)
    return attention_out

# =====================================================================
# STEP 5: Optuna Hyperparameter Optimization
# =====================================================================
def objective(trial):
    """
    Objective function for Optuna to minimize validation loss (MSE)
    by tuning model architecture and training hyperparameters.
    """
    # Hyperparameters to tune
    lstm_units_1 = trial.suggest_categorical('lstm_units_1', [64, 128, 256])
    lstm_units_2 = trial.suggest_categorical('lstm_units_2', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True) # Updated to suggest_float
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    epochs = 20  # Keep epochs relatively low for faster trials

    # Build the dynamic model
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_out = layers.LSTM(lstm_units_1, return_sequences=True)(inputs)
    attention_out = self_attention_block(lstm_out, num_heads=num_heads)
    lstm_out2 = layers.LSTM(lstm_units_2)(attention_out)
    output = layers.Dense(1)(lstm_out2)
    
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    
    # Return the minimum validation loss for Optuna to optimize
    val_loss = min(history.history['val_loss'])
    return val_loss

# Run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40)  # Increase n_trials for more extensive searching

print('Best hyperparameters: ', study.best_params)

# =====================================================================
# STEP 6: Final Model Training & Saving
# =====================================================================
best_params = study.best_params

# Rebuild the model using the optimal hyperparameters found by Optuna
inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm_out = layers.LSTM(best_params['lstm_units_1'], return_sequences=True)(inputs)
attention_out = self_attention_block(lstm_out, num_heads=best_params['num_heads'])
lstm_out2 = layers.LSTM(best_params['lstm_units_2'])(attention_out)
output = layers.Dense(1)(lstm_out2)

final_model = models.Model(inputs=inputs, outputs=output)
final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')

# Train the final model with more epochs on the optimal batch size
history = final_model.fit(X_train, y_train, epochs=50, batch_size=best_params['batch_size'], validation_data=(X_test, y_test))

# Evaluate test loss
test_loss = final_model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Save the optimal model
model_save_path = r"[INSERT_YOUR_MODEL_SAVE_PATH_HERE]"
final_model.save(model_save_path)
print("Model saved successfully!")

# =====================================================================
# STEP 7: Model Evaluation & Visualization
# =====================================================================
print(f"\n🔍 Evaluating Model Performance with Signal: {signal_columns}")

def prepare_evaluation_data(df, signal_col):
    """Similar to prepare_data, but tracks time stamps for visualization."""
    windows, labels, time_stamps = [], [], []
    for i in range(0, len(df) - window_size, stride):
        window = df.iloc[i:i+window_size]
        windows.append(window[[signal_col, 'RUL']].values)
        labels.append(window['RUL'].iloc[-1])
        end_time = (i + window_size - 1) * time_per_point
        time_stamps.append(end_time)
    return np.array(windows), np.array(labels), np.array(time_stamps)

# Prepare sequential test data
test_windows, test_labels, test_time_stamps = prepare_evaluation_data(df, signal_columns[0])

# Load model (using custom_objects to ensure MSE metric is recognized correctly)
model = tf.keras.models.load_model(model_save_path, custom_objects={'mse': tf.keras.metrics.MeanSquaredError()})

# Generate predictions
predictions = model.predict(test_windows).flatten()

# Re-fit scaler to inverse transform RUL values back to actual limits
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(test_labels.reshape(-1, 1))

# Inverse transform to get actual RUL metrics
predictions_rescaled = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
actual_rul_rescaled = scaler_y.inverse_transform(test_labels.reshape(-1, 1)).flatten()

# Calculate Error Metrics
mae = mean_absolute_error(actual_rul_rescaled, predictions_rescaled)
rmse = np.sqrt(mean_squared_error(actual_rul_rescaled, predictions_rescaled))
print(f"✅ MAE: {mae:.4f} | RMSE: {rmse:.4f}")

# Plotting Results
plt.figure(figsize=(12, 6))
plt.plot(test_time_stamps, actual_rul_rescaled, label="Actual RUL", color='blue')

# Apply Savitzky-Golay filter to smooth out the noisy raw predictions
smoothed = savgol_filter(predictions_rescaled, 51, 3)
plt.plot(test_time_stamps, smoothed, label="Smoothed Prediction", color='red')

plt.scatter(test_time_stamps, predictions_rescaled, s=10, color='orange', label="Raw Prediction")
plt.title(f"RUL Prediction vs Actual RUL - {signal_columns[0]}")
plt.xlabel("Time (s)")
plt.ylabel("Remaining Useful Life")
plt.legend()
plt.show()