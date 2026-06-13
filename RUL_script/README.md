# Remaining Useful Life (RUL) Prediction for Rolling Bearings

This repository contains a comprehensive machine learning and deep learning pipeline for predicting the Remaining Useful Life (RUL) of rolling bearings. It processes raw vibration signals, denoises them, labels the data based on degradation metrics, and evaluates multiple predictive models ranging from traditional machine learning to advanced attention-based neural networks.

## 📊 Dataset Requirements

This project is built to process the **XJTU-SY Bearing Dataset** (or similarly structured continuous vibration data). 

**Important:** The raw data is not included in this repository due to size constraints. 
1. Download the dataset to your local machine.
2. Open the scripts and update the `[INSERT_..._PATH_HERE]` placeholder variables with your local directory paths before running.

## 📁 Project Structure & Workflow

The pipeline is divided into several independent, sequential stages:

### 1. Signal Denoising (`LSTM-Autoencoder_processing.py`)
Uses an LSTM-Autoencoder to filter out operational noise from the raw 2-channel (Horizontal and Vertical) vibration signals. 

### 2. Denoising Validation (`FFT_analysis.py`)
Performs a Fast Fourier Transform (FFT) analysis. It compares the frequency spectrums of the signals before and after denoising to validate the performance of the Autoencoder.
> **Note:** This script saves comparative charts. Ensure you create the output destination folder on your machine before running to avoid `FileNotFound` errors from `matplotlib`.

### 3. Full Lifecycle Visualization (`XJTU_analysis.py`)
Provides an interactive, full-lifecycle visualization of the bearing's degradation using `plotly`. 
> **Note:** Because GitHub natively strips interactive JavaScript, running this file locally is highly recommended so you can zoom in on specific signal spikes and degradation trends.

### 4. RUL Labeling (`RUL_label_2.py`)
Calculates the First Predicting Time (FPT) using a $3\sigma$ (three-sigma) statistical threshold to pinpoint where degradation begins, then calculates and bins the RUL. 
> **Customization Tip:** The `T0 = 1000` variable inside the `calculate_fpt` function establishes the "healthy baseline" window. You can tweak this value depending on the sampling rate of your specific dataset.

### 5. Advanced Predictive Modeling (`rul_lstm_with_attention_optimize_by_optuna.py`)
The core deep learning model. It utilizes an LSTM network paired with a Multi-Head Self-Attention mechanism to capture long-term dependencies in the degradation process. It dynamically optimizes hyperparameters (like learning rate, batch size, and LSTM units) using `Optuna`.

### 6. Baseline ML Models (`svr_regressor.py` & `xgboost_regressor.py`)
These scripts train and evaluate traditional Machine Learning models (Support Vector Regression and XGBoost) to serve as performance baselines against the deep learning approach.

---

## 🛠️ Installation & Setup

Ensure you have Python installed, then install the required dependencies. You can install them via pip:

```bash
pip install pandas numpy matplotlib scipy scikit-learn tensorflow xgboost optuna plotly joblib