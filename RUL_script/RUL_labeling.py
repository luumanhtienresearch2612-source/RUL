import numpy as np
import pandas as pd
import os

# =====================================================================
# STEP 1: Global Configuration
# =====================================================================
# Sampling frequency: 32768 data points = 1 minute of recording
sampling_rate = 32768  # points per minute
seconds_per_point = 60 / sampling_rate  # seconds per point

# =====================================================================
# STEP 2: First Predicting Time (FPT) Calculation
# =====================================================================
def calculate_fpt(data, threshold=3):
    """
    Detects the First Predicting Time (FPT) using the 3-sigma method.
    It establishes a baseline from the initial healthy state and flags 
    the point where degradation begins.
    """
    T0 = 1000  # Initial sample range to calculate healthy baseline (can be tuned)
    mu = np.mean(data[:T0])     # Mean of the non-degraded period
    sigma = np.std(data[:T0])   # Standard deviation of the non-degraded period

    # Detect FPT using the 3-sigma rule
    for i in range(1, len(data) - 1):
        # Check if two consecutive points fall outside the 3-sigma range
        if (data[i] > mu + threshold * sigma or data[i] < mu - threshold * sigma) and \
           (data[i - 1] > mu + threshold * sigma or data[i - 1] < mu - threshold * sigma):
            return i  # Return the time index of the FPT
            
    return -1  # Return -1 if no FPT is detected

# =====================================================================
# STEP 3: Remaining Useful Life (RUL) Calculation
# =====================================================================
def calculate_rul(fpt_index, failure_time, time_series_length):
    """
    Calculates the RUL for each data point. RUL is 1.0 (100%) before the FPT.
    After FPT, it decreases linearly towards 0 at the known failure time.
    """
    if fpt_index != -1:
        # Convert FPT index (i) to actual time (minutes)
        fpt_time = fpt_index / sampling_rate  

        # Create a time array from FPT to the end of the series (convert index to real time)
        time_series = np.arange(fpt_index, time_series_length) / sampling_rate  

        # Calculate linearly degrading RUL for points after FPT
        rul = np.maximum(0, (failure_time - time_series) / (failure_time - fpt_time))

        # Assign RUL = 1.0 for all points before FPT, then append the degrading RUL
        return np.concatenate((np.ones(fpt_index), rul))  
    else:
        # If no FPT is detected, assume the bearing is completely healthy (RUL = 1.0)
        return np.ones(time_series_length)

# =====================================================================
# STEP 4: RUL Binning (Resolution Reduction)
# =====================================================================
def bin_rul(rul, bin_size=0.05):
    """
    Bins the RUL values into specified intervals to reduce noise/resolution.
    For example, with bin_size=0.05: values between 0.95-1.00 become 0.95.
    """
    return np.floor(rul / bin_size) * bin_size

def calculate_rul_with_bins(fpt_index, failure_time, time_series_length, bin_size=0.05):
    """
    Wrapper function to calculate RUL and immediately apply binning.
    """
    rul = calculate_rul(fpt_index, failure_time, time_series_length)
    binned_rul = np.array([bin_rul(r, bin_size) for r in rul])
    return binned_rul

# =====================================================================
# STEP 5: Batch Processing & Labeling
# =====================================================================
def label_rul_for_all_files(file_list, bearing_failure_time, bin_size=0.05):
    """
    Iterates through a list of CSV files, calculates RUL based on FPT, 
    and saves new labeled CSV files.
    """
    rul_results = []

    for file in file_list:
        print(f"Processing file: {file}")
        data = pd.read_csv(file)
        
        # Extract the denoised horizontal signal for FPT detection
        denoised_data = data['Denoised_Horizontal'].values
        
        # Extract just the bearing name from the filename (e.g., 'Bearing1_1')
        bearing_name = os.path.basename(file).replace('.csv', '')  
        print(f"Bearing name: {bearing_name}")

        # Look up the actual failure time for this specific bearing
        failure_time = bearing_failure_time.get(bearing_name, None)
        if failure_time is None:
            print(f"⚠️ Failure time for {bearing_name} not found! Skipping...")
            continue

        print(f"Known Failure time: {failure_time} minutes")

        # Calculate FPT
        fpt_index = calculate_fpt(denoised_data)
        print(f"Detected FPT index: {fpt_index}")
        if fpt_index == -1:
            print("⚠️ No FPT detected! Skipping...")
            continue

        # Calculate binned RUL and append it as a new column
        rul = calculate_rul_with_bins(fpt_index, failure_time, len(denoised_data), bin_size)
        data['RUL'] = rul
        rul_results.append(data)

        # Save the updated DataFrame to a new CSV file
        output_file_path = file.replace('.csv', '_RUL_labeled.csv')
        data.to_csv(output_file_path, index=False)
        print(f"✅ Successfully saved labeled file: {output_file_path}\n")

    return rul_results

# =====================================================================
# STEP 6: Execution Example
# =====================================================================
if __name__ == "__main__":
    # Update this list with the paths to your denoised CSV files
    file_list = [
        r"[INSERT_YOUR_DENOISED_FILE_PATH_HERE]/Bearing2_5.csv"
    ]  
    
    # Dictionary containing the actual failure times (in minutes) for each bearing
    bearing_failure_time = {
        'Bearing1_1': 123,  
        'Bearing1_2': 161,
        'Bearing1_3': 158,
        'Bearing1_4': 122,
        'Bearing1_5': 52,
        'Bearing2_1': 491,
        'Bearing2_2': 161,
        'Bearing2_3': 533,
        'Bearing2_4': 42,
        'Bearing2_5': 339,
    }

    # Execute the labeling process
    rul_results = label_rul_for_all_files(file_list, bearing_failure_time)