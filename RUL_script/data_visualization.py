import os
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np

# =====================================================================
# STEP 1: Configuration and Path Setup
# =====================================================================
# Update this path to the directory containing your specific bearing dataset CSVs.
folder_path = r"[INSERT_YOUR_DATASET_DIRECTORY_PATH_HERE]"

# Set the Sampling Frequency (Fs in Hz)
Fs = 25600  # Example: 25.6 kHz (Adjust based on the specific dataset documentation)

# Check if the directory exists to prevent runtime errors
if not os.path.exists(folder_path):
    print("❌ Error: The specified directory does not exist!")
    exit()

# =====================================================================
# STEP 2: Data Loading and Aggregation
# =====================================================================
# Retrieve and sort all CSV files numerically based on their filenames
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')], key=lambda x: int(x.split('.')[0]))

dataframes = []
for file in csv_files:
    file_path = os.path.join(folder_path, file)

    # Read the CSV file (assuming no header in the raw dataset)
    df = pd.read_csv(file_path, header=None)

    # Convert values to numeric, coercing any read errors (like text) into NaN
    df[0] = pd.to_numeric(df[0], errors='coerce')  # Channel 1: Horizontal Vibration
    df[1] = pd.to_numeric(df[1], errors='coerce')  # Channel 2: Vertical Vibration

    # Assign a file ID to keep track of the origin file
    df['file_id'] = int(file.split('.')[0])  
    dataframes.append(df)

# Merge all individual files into a single, continuous timeline DataFrame
full_data = pd.concat(dataframes, ignore_index=True)

# Drop any NaN values caused by the numeric conversion
full_data.dropna(inplace=True)

# =====================================================================
# STEP 3: Downsampling & Time Calculation
# =====================================================================
# Compute the actual continuous time in minutes for the full dataset
full_data['time_min'] = (full_data.index / Fs) / 60  

# Downsampling: Select every 100th data point for memory-efficient rendering
# This is crucial for plotting large time-series data without crashing the browser
sample_rate = 100
sampled_data = full_data.iloc[::sample_rate, :].copy()

# Adjust the time calculation for the downsampled DataFrame
sampled_data['time_min'] = (sampled_data.index * sample_rate / Fs) / 60  

# =====================================================================
# STEP 4: Visualization Setup (Plotly)
# =====================================================================
# Generate 10 evenly spaced time tick values to force uniformity across both graphs
tick_values = np.linspace(sampled_data['time_min'].min(), sampled_data['time_min'].max(), num=10)

# Create Subplots for Horizontal and Vertical Vibration Data
fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=False,  
                       subplot_titles=("Horizontal Vibration Data", "Vertical Vibration Data"))

# Add Horizontal Vibration Signal (Blue)
fig.add_trace(go.Scatter(x=sampled_data['time_min'], y=sampled_data[0], mode='lines',
                         name="Horizontal Vibration", line=dict(color='blue')), row=1, col=1)

# Add Vertical Vibration Signal (Green)
fig.add_trace(go.Scatter(x=sampled_data['time_min'], y=sampled_data[1], mode='lines',
                         name="Vertical Vibration", line=dict(color='green')), row=2, col=1)

# =====================================================================
# STEP 5: Layout Formatting & Display
# =====================================================================
# Set Y-Axis Titles
fig.update_yaxes(title_text="Vibration Amplitude (g)", row=1, col=1)
fig.update_yaxes(title_text="Vibration Amplitude (g)", row=2, col=1)

# Force both graphs to show the exact same time values on the X-Axis
fig.update_xaxes(title_text="Time (Minutes)", tickvals=tick_values, ticktext=[f"{t:.0f}" for t in tick_values], row=1, col=1) 
fig.update_xaxes(title_text="Time (Minutes)", tickvals=tick_values, ticktext=[f"{t:.0f}" for t in tick_values], row=2, col=1) 

# Update Global Layout properties
fig.update_layout(
    title="Bearing Full Lifecycle Vibration Data",
    height=800, 
    showlegend=True
)

# Render the interactive plot
fig.show()