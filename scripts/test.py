#!/usr/bin/env python
# coding: utf-8

"""
Test Script for Temporal CNN Model for SOC Estimation
Author: Usama Yasir Khan
Description: This script tests the trained Temporal CNN model on provided test data
and generates evaluation metrics and visualizations, including residual and error analysis.
"""

import os
import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Function to load and preprocess test data
def load_test_data(file_path, input_columns, target_column, timesteps):
    """Load and preprocess test data for evaluation."""
    mat_file = scipy.io.loadmat(file_path)
    X = mat_file['X'].T
    Y = mat_file['Y'].T
    df_X = pd.DataFrame(X, columns=input_columns)
    df_Y = pd.DataFrame(Y, columns=[target_column])
    df = pd.concat([df_X, df_Y], axis=1)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(df) - timesteps):
        X_seq.append(df[input_columns].iloc[i:i + timesteps].values)
        y_seq.append(df[target_column].iloc[i + timesteps])
    return np.array(X_seq), np.array(y_seq)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, temp_label):
    """Evaluate the model and generate metrics and plots."""
    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"{temp_label} - Mean Absolute Error (MAE): {mae}")
    print(f"{temp_label} - Root Mean Squared Error (RMSE): {rmse}")
    print(f"{temp_label} - R-squared (R²): {r2}")

    # Plot Actual vs Predicted SOC
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual SOC', color='blue')
    plt.plot(y_pred, label='Predicted SOC', color='red', linestyle='--')
    plt.title(f"Actual vs Predicted SOC - {temp_label}")
    plt.xlabel("Samples")
    plt.ylabel("SOC")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/soc_comparison_{temp_label.replace('°', '')}.png")
    plt.show()

    # Plot Residuals (Actual - Predicted)
    residuals = y_test - y_pred.flatten()
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred.flatten(), residuals, alpha=0.6, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.title(f"Residuals vs Predicted SOC - {temp_label}")
    plt.xlabel("Predicted SOC")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.grid(True)
    plt.savefig(f"results/soc_residuals_{temp_label.replace('°', '')}.png")
    plt.show()

    # Plot Prediction Error (Absolute Error)
    prediction_error = np.abs(residuals)
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_error, label='Prediction Error', color='purple')
    plt.title(f"Prediction Error Over Samples - {temp_label}")
    plt.xlabel("Samples")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/prediction_error_{temp_label.replace('°', '')}.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Define test files and model path
    test_files = {
        "-10°C": "Test/02_TEST_LGHG2@-10degC_Norm_(05_Inputs).mat",
        "0°C": "Test/02_TEST_LGHG2@0degC_Norm_(05_Inputs).mat",
        "10°C": "Test/02_TEST_LGHG2@10degC_Norm_(05_Inputs).mat",
        "25°C": "Test/02_TEST_LGHG2@25degC_Norm_(05_Inputs).mat"
    }
    model_path = "updated_cnn_model.keras"
    input_columns = ['Voltage', 'Current', 'Temperature', 'Avg_voltage', 'Avg_current']
    target_column = 'SOC'
    timesteps = 30

    # Load the trained model
    print("Loading the trained Temporal CNN model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Evaluate on each test dataset
    for temp_label, file_path in test_files.items():
        if os.path.exists(file_path):
            print(f"\nEvaluating model on {temp_label} data...")
            X_test, y_test = load_test_data(file_path, input_columns, target_column, timesteps)
            evaluate_model(model, X_test, y_test, temp_label)
        else:
            print(f"Test file for {temp_label} not found. Skipping.")
