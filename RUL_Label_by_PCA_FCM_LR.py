import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu từ tệp
file_path = r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN_denoise_by_LSTM_Autoencoder\Bearing1_2.csv"
data = pd.read_csv(file_path)

# Chọn các đặc trưng Denoised_Horizontal và Denoised_Vertical
features = data[['Denoised_Horizontal', 'Denoised_Vertical']]

# 1. PCA - Giảm chiều dữ liệu và tạo degradation index
pca = PCA(n_components=1)
degradation_index = pca.fit_transform(features)  # Lấy chỉ số thoái hóa từ PCA

# Vẽ biểu đồ degradation index
plt.plot(degradation_index, label='Degradation Index')
plt.title('Degradation Index by PCA')
plt.legend()
plt.show()

# 2. FCM - Phân cụm dữ liệu và xác định các điểm chuyển tiếp
n_clusters = 3  # Số cụm: Normal, Slight Fault, Heavy Fault

# Áp dụng FCM để phân cụm
degradation_index = degradation_index.reshape(-1, 1)  # Đảm bảo dữ liệu là mảng 2D
cntr, u, _, _, _, _, _ = fuzz.cmeans(degradation_index.T, c=n_clusters, m=2, error=0.005, maxiter=1000)

# Xác định các điểm chuyển tiếp (turning points) từ độ hội tụ
# Chúng ta sẽ tìm điểm nào có độ hội tụ lớn nhất cho mỗi mẫu
turning_points = np.argmax(u, axis=0)

# Vẽ kết quả FCM
plt.plot(degradation_index, label='Degradation Index')
for i in range(1, n_clusters):  # Vẽ các điểm chuyển tiếp
    plt.scatter(np.where(turning_points == i)[0], degradation_index[turning_points == i], color='red', label=f'Turning Point {i}')
plt.title('FCM Clustering Results')
plt.legend()
plt.show()

# 3. Linear Regression - Tạo nhãn RUL lý tưởng từ degradation index
X = np.arange(len(degradation_index)).reshape(-1, 1)  # Tạo trục thời gian
y = degradation_index  # Chỉ số thoái hóa làm nhãn mục tiêu

# Áp dụng hồi quy tuyến tính
regressor = LinearRegression()
regressor.fit(X, y)

# Dự đoán RUL lý tưởng
rul_ideal = regressor.predict(X)

# Vẽ kết quả Linear Regression
plt.plot(X, y, label='Degradation Index')
plt.plot(X, rul_ideal, label='Ideal RUL Label', linestyle='--')
plt.title('Linear Regression for RUL Label')
plt.legend()
plt.show()

# Lưu kết quả RUL lý tưởng nếu cần
data['RUL_Ideal'] = rul_ideal
data.to_csv(r"D:\XJTU-SY_Bearing_Datasets\RUL_Labels\RUL_Ideal_Bearing1_2.csv", index=False)

# In ra thông tin về kết quả
print("RUL ideal has been saved to the specified path.")
