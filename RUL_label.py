import numpy as np
import pandas as pd
import os

# Tần số lấy mẫu: 32768 điểm dữ liệu = 1 phút
sampling_rate = 32768  # points per minute
seconds_per_point = 60 / sampling_rate  # seconds per point


# Hàm tính toán FPT sử dụng phương pháp 3σ
def calculate_fpt(data, threshold=3):
    T0 = 1000  # Chọn khoảng ban đầu để tính trung bình (có thể thay đổi)
    mu = np.mean(data[:T0])  # Trung bình của khoảng không suy giảm
    sigma = np.std(data[:T0])  # Độ lệch chuẩn

    # Phát hiện FPT bằng phương pháp 3σ
    for i in range(1, len(data) - 1):
        # Kiểm tra xem hai điểm liên tiếp có nằm ngoài phạm vi 3σ không
        if (data[i] > mu + threshold * sigma or data[i] < mu - threshold * sigma) and \
           (data[i - 1] > mu + threshold * sigma or data[i - 1] < mu - threshold * sigma):
            return i  # Trả về chỉ số thời gian của FPT
    return -1  # Nếu không phát hiện FPT


# Hàm tính toán RUL từ FPT
def calculate_rul(fpt_index, failure_time, time_series_length):
    if fpt_index != -1:
        # Quy đổi FPT (i) thành thời gian thực tế (phút)
        fpt_time = fpt_index  / sampling_rate  # Thời gian FPT (phút)

        # Tạo mảng thời gian từ FPT đến cuối chuỗi (chỉ số thành thời gian thực)
        time_series = np.arange(fpt_index, time_series_length) / sampling_rate # chuyển chỉ số thành thời gian (phút)

        # Tính toán RUL cho từng điểm thời gian
        rul = np.maximum(0, (failure_time - time_series) / (failure_time - fpt_time))

        # Gán RUL = 1 cho tất cả các điểm trước FPT
        return np.concatenate((np.ones(fpt_index), rul))  # Gán 1 cho tất cả các điểm trước FPT
    else:
        # Nếu không phát hiện FPT, gán toàn bộ là 1
        return np.ones(time_series_length)

def label_rul_for_all_files(file_list, bearing_failure_time):
    rul_results = []

    for file in file_list:
        print(f"Processing file: {file}")
        data = pd.read_csv(file)
        print(f"Data loaded for {file}: {data.head()}")  # Print first few rows

        denoised_data = data['Denoised_Horizontal'].values
        bearing_name = os.path.basename(file).replace('.csv', '')  # Extract just the bearing name
        print(f"Bearing name: {bearing_name}")

        failure_time = bearing_failure_time.get(bearing_name, None)
        if failure_time is None:
            print(f"Failure time for {bearing_name} not found!")
            continue

        print(f"Failure time: {failure_time}")

        fpt_index = calculate_fpt(denoised_data)
        print(f"FPT index: {fpt_index}")
        if fpt_index == -1:
            print("No FPT detected!")
            continue

        rul = calculate_rul(fpt_index, failure_time, len(denoised_data))
        data['RUL'] = rul
        rul_results.append(data)

        output_file_path = file.replace('.csv', '_RUL_labeled_14_09.csv')
        data.to_csv(output_file_path, index=False)
        print(f"Đã lưu file: {output_file_path} với RUL labels")

    return rul_results


# Ví dụ về cách sử dụng:
file_list = [
             "D:\XJTU-SY_Bearing_Datasets\\37.5Hz11kN_denoise_by_LSTM_Autoencoder\\Bearing2_4.csv"]  # Cập nhật đường dẫn file của bạn
bearing_failure_time = {
    'Bearing1_2': 161,  # 2h 3m = 123 phút
    'Bearing2_2': 161,
    'Bearing2_1':491,
    'Bearing2_3':533,
    'Bearing2_4':42,
    'Bearing2_5':339,
    'Bearing1_3':158,
    'Bearing1_4':122,
    'Bearing1_5':52,
    'Bearing1_1':123,
}

# Gán nhãn RUL cho tất cả các tệp dữ liệu
rul_results = label_rul_for_all_files(file_list, bearing_failure_time)
