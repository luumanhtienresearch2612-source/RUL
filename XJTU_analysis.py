import os
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np

# ✅ Define the folder path correctly
folder_path = r"D:\XJTU-SY_Bearing_Datasets\40Hz10kN\Bearing3_1"

# ✅ Set the Sampling Frequency (Fs in Hz)
Fs = 25600  # Example: 25.6 kHz (Adjust based on dataset documentation)

# ✅ Check if the directory exists
if not os.path.exists(folder_path):
    print("❌ Lỗi: Đường dẫn không tồn tại!")
    exit()

# ✅ Get all CSV files in sorted order
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')], key=lambda x: int(x.split('.')[0]))

# ✅ Read and concatenate all CSV files
dataframes = []
for file in csv_files:
    file_path = os.path.join(folder_path, file)

    # Ensure proper reading (Check if the first row is a header)
    df = pd.read_csv(file_path, header=None)

    # Convert values to numeric (in case they are read as strings)
    df[0] = pd.to_numeric(df[0], errors='coerce')  # Horizontal Vibration
    df[1] = pd.to_numeric(df[1], errors='coerce')  # Vertical Vibration

    df['file_id'] = int(file.split('.')[0])  # Assign file ID
    dataframes.append(df)

# ✅ Merge all dataframes
full_data = pd.concat(dataframes, ignore_index=True)

# ✅ Drop any NaN values caused by type conversion issues
full_data.dropna(inplace=True)

# ✅ Compute Time in Minutes
full_data['time_min'] = (full_data.index / Fs) / 60  # Convert to minutes

# ✅ Downsampling: Select every 100th data point for visualization
sample_rate = 100
sampled_data = full_data.iloc[::sample_rate, :].copy()

# ✅ Adjust Time for Downsampling (Multiply Index by Sample Rate)
sampled_data['time_min'] = (sampled_data.index * sample_rate / Fs) / 60  # Convert to minutes

# ✅ Generate Time Tick Values (force same tick values for both graphs)
tick_values = np.linspace(sampled_data['time_min'].min(), sampled_data['time_min'].max(), num=10)

# ✅ Create Subplots for Horizontal and Vertical Vibration Data
fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=False,  # 🚨 DISABLE SHARED X-AXIS 🚨
                       subplot_titles=("Horizontal Vibration Data", "Vertical Vibration Data"))

# ✅ Add Horizontal Vibration Signal (Blue)
fig.add_trace(go.Scatter(x=sampled_data['time_min'], y=sampled_data[0], mode='lines',
                         name="Horizontal Vibration", line=dict(color='blue')), row=1, col=1)

# ✅ Add Vertical Vibration Signal (Green)
fig.add_trace(go.Scatter(x=sampled_data['time_min'], y=sampled_data[1], mode='lines',
                         name="Vertical Vibration", line=dict(color='green')), row=2, col=1)

# ✅ Set Y-Axis Titles
fig.update_yaxes(title_text="Vibration Amplitude (g)", row=1, col=1)
fig.update_yaxes(title_text="Vibration Amplitude (g)", row=2, col=1)

# ✅ Force Both Graphs to Show the Same Time Values
fig.update_xaxes(title_text="Time (Minutes)", tickvals=tick_values, ticktext=[f"{t:.0f}" for t in tick_values], row=1, col=1)  # First Graph Time Row
fig.update_xaxes(title_text="Time (Minutes)", tickvals=tick_values, ticktext=[f"{t:.0f}" for t in tick_values], row=2, col=1)  # Second Graph Time Row

# ✅ Update Layout
fig.update_layout(
    title="Vibration Data: 40Hz10kN_Bearing3_1",
    height=800, showlegend=True
)

fig.show()
