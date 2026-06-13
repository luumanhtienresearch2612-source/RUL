import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler
import os

# ----------- CONFIGURATION -----------
fs = 25600  # Sampling frequency
before_path = r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_2"
after_path = r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN_denoise_by_LSTM_Autoencoder\Bearing1_2.csv"
before_files = [os.path.join(before_path, f"{i}.csv") for i in range(1, 161)]

# ----------- DATA LOADING -----------
df_before = pd.concat([pd.read_csv(f) for f in before_files], ignore_index=True)
df_after = pd.read_csv(after_path)

print("Before denoising:", df_before.shape)
print("After denoising :", df_after.shape)

# ----------- INVERSE SCALING & REMOVE DC OFFSET -----------
scaler = MinMaxScaler()
scaler.fit(df_before[['Horizontal_vibration_signals', 'Vertical_vibration_signals']])
denoised_scaled = df_after[['Denoised_Horizontal', 'Denoised_Vertical']].values
denoised_inverse = scaler.inverse_transform(denoised_scaled)
df_after['Denoised_Horizontal'] = denoised_inverse[:, 0] - denoised_inverse[:, 0].mean()
df_after['Denoised_Vertical'] = denoised_inverse[:, 1] - denoised_inverse[:, 1].mean()

# ----------- UTILS -----------
def get_time_axis(num_samples):
    return np.arange(num_samples) / 32768  # 32768 samples per minute

def compute_fft(signal, fs):
    signal = signal - np.mean(signal)
    N = len(signal)
    T = 1.0 / fs
    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]
    amplitude = 2.0 / N * np.abs(yf[0:N // 2])
    return xf, amplitude

# ----------- COMBINED VISUALIZATION -----------

fig, axs = plt.subplots(4, 2, figsize=(16, 12))

# --- Horizontal Before ---
time_h_before = get_time_axis(len(df_before))
axs[0, 0].plot(time_h_before, df_before['Horizontal_vibration_signals'], label='Horizontal Before Denoising')
axs[0, 0].set_title("Horizontal Vibration Signal - Before Denoising")
axs[0, 0].set_xlabel("Time (minutes)")
axs[0, 0].set_ylabel("Amplitude")

xf, amp = compute_fft(df_before['Horizontal_vibration_signals'], fs)
axs[0, 1].plot(xf, amp)
axs[0, 1].set_title("FFT of Horizontal Before Denoising")
axs[0, 1].set_xlabel("Frequency (Hz)")
axs[0, 1].set_ylabel("Amplitude")

# --- Horizontal After ---
time_h_after = get_time_axis(len(df_after))
axs[1, 0].plot(time_h_after, df_after['Denoised_Horizontal'], color='green', label='Horizontal After Denoising')
axs[1, 0].set_title("Horizontal Vibration Signal - After Denoising")
axs[1, 0].set_xlabel("Time (minutes)")
axs[1, 0].set_ylabel("Amplitude")

xf, amp = compute_fft(df_after['Denoised_Horizontal'], fs)
axs[1, 1].plot(xf, amp)
axs[1, 1].set_title("FFT of Horizontal After Denoising")
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_ylabel("Amplitude")

# --- Vertical Before ---
time_v_before = get_time_axis(len(df_before))
axs[2, 0].plot(time_v_before, df_before['Vertical_vibration_signals'], label='Vertical Before Denoising')
axs[2, 0].set_title("Vertical Vibration Signal - Before Denoising")
axs[2, 0].set_xlabel("Time (minutes)")
axs[2, 0].set_ylabel("Amplitude")

xf, amp = compute_fft(df_before['Vertical_vibration_signals'], fs)
axs[2, 1].plot(xf, amp)
axs[2, 1].set_title("FFT of Vertical Before Denoising")
axs[2, 1].set_xlabel("Frequency (Hz)")
axs[2, 1].set_ylabel("Amplitude")

# --- Vertical After ---
time_v_after = get_time_axis(len(df_after))
axs[3, 0].plot(time_v_after, df_after['Denoised_Vertical'], color='green', label='Vertical After Denoising')
axs[3, 0].set_title("Vertical Vibration Signal - After Denoising")
axs[3, 0].set_xlabel("Time (minutes)")
axs[3, 0].set_ylabel("Amplitude")

xf, amp = compute_fft(df_after['Denoised_Vertical'], fs)
axs[3, 1].plot(xf, amp)
axs[3, 1].set_title("FFT of Vertical After Denoising")
axs[3, 1].set_xlabel("Frequency (Hz)")
axs[3, 1].set_ylabel("Amplitude")

plt.tight_layout()
output_path = r"D:\XJTU-SY_Bearing_Datasets\visualize_data\35Hz12kN\FFT_Analysis\bearing1_2\all_signals_fft.png"
plt.savefig(output_path)
plt.show()

print(f"✅ Combined figure saved at: {output_path}")
