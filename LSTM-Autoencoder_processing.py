import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

# 📌 Đọc và ghép nối toàn bộ file CSV thành một chuỗi liên tục
data_path = r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_1"  # Use raw string format
all_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')])

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

full_data = pd.concat(df_list, ignore_index=True)

# 📌 Lấy cả hai cột dữ liệu Horizontal & Vertical
time_steps = 100  # Chia tín hiệu thành đoạn 100 bước thời gian
features = 2  # Có hai tín hiệu: Horizontal & Vertical

# Chia tín hiệu thành đoạn có độ dài time_steps
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i : i + time_steps])
    return np.array(sequences)

# Xử lý cả hai cột: Horizontal và Vertical
X = create_sequences(full_data[['Horizontal_vibration_signals', 'Vertical_vibration_signals']].values, time_steps)

# 📌 Xây dựng LSTM-Autoencoder xử lý cả hai tín hiệu
input_dim = (time_steps, features)

inputs = tf.keras.Input(shape=input_dim)
encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)
encoded = RepeatVector(time_steps)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(features))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

# 📌 Huấn luyện Autoencoder trên toàn bộ dữ liệu (KHÔNG SỬ DỤNG FPT)
autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.1)

# 📌 Sử dụng Autoencoder để lọc nhiễu trên cả hai tín hiệu
X_denoised = autoencoder.predict(X)

# 📌 Visualize dữ liệu gốc và dữ liệu đã khử nhiễu
plt.figure(figsize=(12, 6))
plt.plot(X[0, :, 0], label="Original Horizontal Vibration", alpha=0.6)
plt.plot(X_denoised[0, :, 0], label="Denoised Horizontal Vibration", linestyle='dashed')
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Vibration Amplitude")
plt.title("Comparison of Original and Denoised Signal (Horizontal)")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(X[0, :, 1], label="Original Vertical Vibration", alpha=0.6)
plt.plot(X_denoised[0, :, 1], label="Denoised Vertical Vibration", linestyle='dashed')
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Vibration Amplitude")
plt.title("Comparison of Original and Denoised Signal (Vertical)")
plt.show()
