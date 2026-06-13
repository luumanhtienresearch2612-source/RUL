import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

# =====================================================================
# STEP 1: Load and Aggregate Data
# =====================================================================
# Update this path to the directory containing your bearing dataset CSVs.
data_path = r"[INSERT_YOUR_DATASET_DIRECTORY_PATH_HERE]"  

# Retrieve and sort all CSV files in the specified directory
all_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')])

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate all individual files into a single, continuous continuous timeline
full_data = pd.concat(df_list, ignore_index=True)

# =====================================================================
# STEP 2: Data Preprocessing & Sequencing
# =====================================================================
time_steps = 1000  # Number of time steps per sequence
features = 2      # Two vibration channels: Horizontal & Vertical

def create_sequences(data, time_steps):
    """
    Splits the continuous time-series data into overlapping sequences 
    using a sliding window approach for LSTM input.
    """
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i : i + time_steps])
    return np.array(sequences)

# Extract the relevant columns and create sequences of shape (samples, time_steps, features)
X = create_sequences(full_data[['Horizontal_vibration_signals', 'Vertical_vibration_signals']].values, time_steps)

# =====================================================================
# STEP 3: Build the LSTM-Autoencoder Architecture
# =====================================================================
input_dim = (time_steps, features)

# --- Encoder ---
inputs = tf.keras.Input(shape=input_dim)
encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)

# --- Bottleneck ---
# Repeat the encoded feature vector across the time steps
encoded = RepeatVector(time_steps)(encoded)

# --- Decoder ---
decoded = LSTM(32, activation='relu', return_sequences=True)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(features))(decoded)

# Compile the Autoencoder model
autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

# =====================================================================
# STEP 4: Model Training
# =====================================================================
# Train the model to reconstruct the original input (X to X)
# Note: FPT (Fast Fourier Transform) preprocessing is bypassed here in favor of raw signals
autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.1)

# =====================================================================
# STEP 5: Signal Denoising
# =====================================================================
# Pass the original noisy sequences through the trained autoencoder to filter noise
X_denoised = autoencoder.predict(X)

# =====================================================================
# STEP 6: Visualization
# =====================================================================

# Plot 1: Original vs. Denoised Horizontal Vibration Signal
plt.figure(figsize=(12, 6))
plt.plot(X[0, :, 0], label="Original Horizontal Vibration", alpha=0.6)
plt.plot(X_denoised[0, :, 0], label="Denoised Horizontal Vibration", linestyle='dashed')
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Vibration Amplitude")
plt.title("Comparison of Original and Denoised Signal (Horizontal)")
plt.show()

# Plot 2: Original vs. Denoised Vertical Vibration Signal
plt.figure(figsize=(12, 6))
plt.plot(X[0, :, 1], label="Original Vertical Vibration", alpha=0.6)
plt.plot(X_denoised[0, :, 1], label="Denoised Vertical Vibration", linestyle='dashed')
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Vibration Amplitude")
plt.title("Comparison of Original and Denoised Signal (Vertical)")
plt.show()