
import pandas as pd
import matplotlib.pyplot as plt

# File path (update if necessary)
file_path = r"D:\XJTU-SY_Bearing_Datasets\35Hz12kN_denoise_by_LSTM_Autoencoder\Bearing1_2_RUL_labeled.csv"

# Load dataset
df = pd.read_csv(file_path)

# Check if 'RUL' column exists
if 'RUL' not in df.columns:
    print("Error: 'RUL' column not found in the dataset!")
else:
    # Plot RUL over time (assuming row index represents time)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['RUL'], label="Remaining Useful Life (RUL)", color='blue')

    # Labels and Title
    plt.xlabel("Time Step")
    plt.ylabel("RUL")
    plt.ylabel("RUL")
    plt.title("Remaining Useful Life (RUL) over Time")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

