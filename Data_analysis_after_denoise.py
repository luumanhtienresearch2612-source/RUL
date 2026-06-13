import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Đọc dữ liệu trước khi khử nhiễu (nhiều file CSV trong thư mục)
before_path = r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_1"
before_files = [f"{before_path}\\{i}.csv" for i in range(1, 123)]  # Chỉnh lại theo đúng số file CSV trong thư mục

# Dữ liệu sau khi khử nhiễu
after_path = r"D:/Downloads/Bearing1_1_denoise_2.csv"

# Đọc dữ liệu trước khi khử nhiễu
# Checking the individual file sizes before concatenating
for f in before_files:
    temp_df = pd.read_csv(f)
    print(f"File: {f}, Rows: {temp_df.shape[0]}")
df_before = pd.concat([pd.read_csv(f) for f in before_files], ignore_index=True)

# Đọc dữ liệu sau khi khử nhiễu
df_after = pd.read_csv(after_path)
# In ra kích thước và số lượng của tập dữ liệu trước khi khử nhiễu
print("Size of the dataset before denoising:")
print(f"Number of samples: {df_before.shape[0]}")
print(f"Number of features: {df_before.shape[1]}")
print()

# In ra kích thước và số lượng của tập dữ liệu sau khi khử nhiễu
print("Size of the dataset after denoising:")
print(f"Number of samples: {df_after.shape[0]}")
print(f"Number of features: {df_after.shape[1]}")
print("Shape of df_after:")
print(df_after.shape)
print(df_after.head())



# Load the dataset after denoising
df_after = pd.read_csv(r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN_denoise_by_LSTM_Autoencoder\Bearing1_1.csv")

# Print the actual shape
print(df_after.shape)




# Vẽ tín hiệu và FFT cho tín hiệu horizontal trước khi khử nhiễu
plt.figure(figsize=(12, 6))

# Vẽ tín hiệu horizontal trước khi khử nhiễu
plt.subplot(2, 1, 1)
plt.plot(df_before['Horizontal_vibration_signals'], label='Horizontal Before Denoising')
plt.title('Before Denoising - Horizontal Vibration Signal')
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.legend()

# Vẽ FFT của tín hiệu horizontal trước khi khử nhiễu
def plot_fft(signal, label, subplot_position):
    N = len(signal)
    T = 1.0 / 8000.0  # Giả sử tần số lấy mẫu là 8kHz (thay đổi nếu cần)
    x = np.linspace(0.0, N*T, N, endpoint=False)
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]
    plt.subplot(2, 1, subplot_position)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.title(f'FFT of {label}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

# FFT của tín hiệu horizontal trước khi khử nhiễu
plot_fft(df_before['Horizontal_vibration_signals'], 'Horizontal Before Denoising', 2)

# Lưu và hiển thị đồ thị cho tín hiệu horizontal
plt.tight_layout()
plt.savefig(r"D:\XJTU-SY_Bearing_Datasets\visualize_data\35Hz12kN\bearing1_1\horizontal_before_comparison.png")
plt.show()

# Vẽ tín hiệu và FFT cho tín hiệu vertical trước khi khử nhiễu
plt.figure(figsize=(12, 6))

# Vẽ tín hiệu vertical trước khi khử nhiễu
plt.subplot(2, 1, 1)
plt.plot(df_before['Vertical_vibration_signals'], label='Vertical Before Denoising')
plt.title('Before Denoising - Vertical Vibration Signal')
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.legend()

# Vẽ FFT của tín hiệu vertical trước khi khử nhiễu
plot_fft(df_before['Vertical_vibration_signals'], 'Vertical Before Denoising', 2)

# Lưu và hiển thị đồ thị cho tín hiệu vertical
plt.tight_layout()
plt.savefig(r"D:\XJTU-SY_Bearing_Datasets\visualize_data\35Hz12kN\bearing1_1\vertical_before_comparison.png")
plt.show()

# Vẽ tín hiệu và FFT cho tín hiệu horizontal sau khi khử nhiễu
plt.figure(figsize=(12, 6))

# Vẽ tín hiệu horizontal sau khi khử nhiễu
plt.subplot(2, 1, 1)
plt.plot(df_after['Denoised_Horizontal'], label='Horizontal After Denoising')
plt.title('After Denoising - Horizontal Vibration Signal')
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.legend()

# Vẽ FFT của tín hiệu horizontal sau khi khử nhiễu
plot_fft(df_after['Denoised_Horizontal'], 'Denoised Horizontal', 2)

# Lưu và hiển thị đồ thị cho tín hiệu horizontal sau khi khử nhiễu
plt.tight_layout()
plt.savefig(r"D:\XJTU-SY_Bearing_Datasets\visualize_data\35Hz12kN\bearing1_4\horizontal_after_comparison.png")
plt.show()

# Vẽ tín hiệu và FFT cho tín hiệu vertical sau khi khử nhiễu
plt.figure(figsize=(12, 6))

# Vẽ tín hiệu vertical sau khi khử nhiễu
plt.subplot(2, 1, 1)
plt.plot(df_after['Denoised_Vertical'], label='Vertical After Denoising')
plt.title('After Denoising - Vertical Vibration Signal')
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.legend()

# Vẽ FFT của tín hiệu vertical sau khi khử nhiễu
plot_fft(df_after['Denoised_Vertical'], 'Denoised Vertical', 2)

# Lưu và hiển thị đồ thị cho tín hiệu vertical sau khi khử nhiễu
plt.tight_layout()
plt.savefig(r"D:\XJTU-SY_Bearing_Datasets\visualize_data\35Hz12kN\bearing1_4\vertical_after_comparison.png")
plt.show()
