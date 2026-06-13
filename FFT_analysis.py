import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# ----------- CONFIGURATION -----------

fs = 25600  # Sampling frequency
before_path = r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_2"
after_path = r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN_denoise_by_LSTM_Autoencoder\Bearing1_2.csv"
before_files = [os.path.join(before_path, f"{i}.csv") for i in range(1, 123)]

# ----------- DATA LOADING -----------

df_before = pd.concat([pd.read_csv(f) for f in before_files], ignore_index=True)
df_after = pd.read_csv(after_path)

print("Before denoising:", df_before.shape)
print("After denoising :", df_after.shape)

# ----------- UPDATED FFT FUNCTION (GỢI Ý 1 + GỢI Ý 3) -----------

def plot_fft(signal, fs, title, subplot_position):
    # Gợi ý 1: Remove DC component
    signal = signal - np.mean(signal)

    N = len(signal)
    T = 1.0 / fs
    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]
    amplitude = 2.0 / N * np.abs(yf[0:N // 2])

    plt.subplot(2, 1, subplot_position)
    plt.plot(xf, amplitude)
    plt.title(f'FFT of {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    # Gợi ý 3: Skip 0 Hz (DC component) when finding dominant frequency
    dominant_idx = np.argmax(amplitude[1:]) + 1
    return xf[dominant_idx], amplitude[dominant_idx]

# ----------- VISUALIZATION -----------

def visualize(signal_before, signal_after, axis_label, save_prefix):
    # Plot before denoising
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal_before, label=f'{axis_label} Before Denoising', alpha=0.7)
    plt.title(f'{axis_label} Vibration Signal - Before Denoising')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    freq_b, amp_b = plot_fft(signal_before, fs, f'{axis_label} Before Denoising', 2)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{axis_label.lower()}_before.png")
    plt.show()

    # Plot after denoising
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal_after, label=f'{axis_label} After Denoising', color='green')
    plt.title(f'{axis_label} Vibration Signal - After Denoising')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    freq_a, amp_a = plot_fft(signal_after, fs, f'{axis_label} After Denoising', 2)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{axis_label.lower()}_after.png")
    plt.show()

    # Kết quả chính
    print(f"📈 {axis_label} Axis:")
    print(f"  - Dominant Frequency BEFORE Denoising: {freq_b:.2f} Hz, Amplitude: {amp_b:.5f}")
    print(f"  - Dominant Frequency AFTER  Denoising: {freq_a:.2f} Hz, Amplitude: {amp_a:.5f}")
    print("-" * 60)

# ----------- APPLY TO BOTH AXES -----------

visualize(
    df_before['Horizontal_vibration_signals'],
    df_after['Denoised_Horizontal'],
    'Horizontal',
    'D:/XJTU-SY_Bearing_Datasets/visualize_data/35Hz12kN/FFT_Analysis/bearing1_1'
)

visualize(
    df_before['Vertical_vibration_signals'],
    df_after['Denoised_Vertical'],
    'Vertical',
    'D:/XJTU-SY_Bearing_Datasets/visualize_data/35Hz12kN/FFT_Analysis/bearing1_1'
)
