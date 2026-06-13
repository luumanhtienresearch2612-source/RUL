import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# =====================================================================
# STEP 1: Configuration and Path Setup
# =====================================================================
fs = 25600  # Sampling frequency (Hz)

# Replace these placeholders with your actual local or repository paths
before_path = r"[INSERT_RAW_DATA_DIRECTORY_PATH_HERE]"
after_path = r"[INSERT_DENOISED_DATA_FILE_PATH_HERE]"

# Generate a list of all raw CSV files to be concatenated
before_files = [os.path.join(before_path, f"{i}.csv") for i in range(1, 123)]

# =====================================================================
# STEP 2: Data Loading
# =====================================================================
# Read and concatenate all raw data files into a single DataFrame
df_before = pd.concat([pd.read_csv(f) for f in before_files], ignore_index=True)

# Read the single CSV file containing the denoised signals
df_after = pd.read_csv(after_path)

print("Shape of data before denoising:", df_before.shape)
print("Shape of data after denoising :", df_after.shape)

# =====================================================================
# STEP 3: Fast Fourier Transform (FFT) Function
# =====================================================================
def plot_fft(signal, fs, title, subplot_position):
    """
    Computes and plots the FFT of a given signal.
    Returns the dominant frequency and its amplitude.
    """
    # Remove the DC component (0 Hz offset) by subtracting the mean
    signal = signal - np.mean(signal)

    N = len(signal)
    T = 1.0 / fs
    
    # Compute the 1D discrete Fourier Transform
    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]
    
    # Calculate the amplitude spectrum
    amplitude = 2.0 / N * np.abs(yf[0:N // 2])

    # Plot the frequency spectrum
    plt.subplot(2, 1, subplot_position)
    plt.plot(xf, amplitude)
    plt.title(f'FFT of {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    # Find the dominant frequency
    # We skip the first index (0 Hz) to ensure we find the actual operational frequency peak
    dominant_idx = np.argmax(amplitude[1:]) + 1
    
    return xf[dominant_idx], amplitude[dominant_idx]

# =====================================================================
# STEP 4: Visualization and Comparison Function
# =====================================================================
def visualize(signal_before, signal_after, axis_label, save_prefix):
    """
    Plots the time-domain and frequency-domain (FFT) signals for both
    the original and denoised data, then saves the figures.
    """
    # --- Plot 1: Before Denoising ---
    plt.figure(figsize=(12, 6))
    
    # Time-domain subplot
    plt.subplot(2, 1, 1)
    plt.plot(signal_before, label=f'{axis_label} Before Denoising', alpha=0.7)
    plt.title(f'{axis_label} Vibration Signal - Before Denoising')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # Frequency-domain (FFT) subplot
    freq_b, amp_b = plot_fft(signal_before, fs, f'{axis_label} Before Denoising', 2)
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{axis_label.lower()}_before.png")
    plt.show()

    # --- Plot 2: After Denoising ---
    plt.figure(figsize=(12, 6))
    
    # Time-domain subplot
    plt.subplot(2, 1, 1)
    plt.plot(signal_after, label=f'{axis_label} After Denoising', color='green')
    plt.title(f'{axis_label} Vibration Signal - After Denoising')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # Frequency-domain (FFT) subplot
    freq_a, amp_a = plot_fft(signal_after, fs, f'{axis_label} After Denoising', 2)
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{axis_label.lower()}_after.png")
    plt.show()

    # --- Print Main Results ---
    print(f"📈 {axis_label} Axis Analysis:")
    print(f"  - Dominant Frequency BEFORE Denoising: {