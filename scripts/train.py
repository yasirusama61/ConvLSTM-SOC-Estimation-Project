"""
train.py

This script trains a ConvLSTM-based model for State of Charge (SOC) estimation in Lithium-ion batteries.
The ConvLSTM architecture captures spatio-temporal dependencies in SOC data, offering enhanced accuracy.
The script includes data loading, sequence creation, model training, and evaluation.

Author: Usama Yasir Khan
Date: 2024-11-19
"""

import os
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Function to load .mat file and return input and target DataFrames
def load_mat_file(file_path, input_columns, target_column):
    mat_file = scipy.io.loadmat(file_path)
    X = mat_file['X'].T
    Y = mat_file['Y'].T
    df_X = pd.DataFrame(X, columns=input_columns)
    df_Y = pd.DataFrame(Y, columns=[target_column])
    return pd.concat([df_X, df_Y], axis=1)

# Function to create sequences from the data
def create_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq), np.array(y_seq)

# Load and preprocess data
train_file = 'TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat'
validation_file = '01_TEST_LGHG2@n10degC_Norm_(05_Inputs).mat'
input_columns = ['Voltage', 'Current', 'Temperature', 'Avg_voltage', 'Avg_current']
target_column = 'SOC'

# Load training and validation data
df_train = load_mat_file(train_file, input_columns, target_column)
df_val = load_mat_file(validation_file, input_columns, target_column)

# Split data into features and target
X_train, y_train = df_train[input_columns], df_train[target_column]
X_val, y_val = df_val[input_columns], df_val[target_column]

# Define timesteps for sequence creation
timesteps = 100

# Create sequences for training and validation
X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, timesteps)
X_val_seq, y_val_seq = create_sequences(X_val.values, y_val.values, timesteps)

# Reshape data for ConvLSTM2D
X_train_conv = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1, 1, X_train_seq.shape[2]))
X_val_conv = X_val_seq.reshape((X_val_seq.shape[0], X_val_seq.shape[1], 1, 1, X_val_seq.shape[2]))

# Build the ConvLSTM model with Global Average Pooling
model = Sequential()

# ConvLSTM layer for spatio-temporal feature extraction
model.add(ConvLSTM2D(filters=32, kernel_size=(1, 2), activation='relu',
                     input_shape=(X_train_conv.shape[1], 1, 1, X_train_conv.shape[4]),
                     return_sequences=True, padding='same'))

# Dropout for regularization
model.add(Dropout(0.3))

# Global Average Pooling layer for dimensionality reduction and robustness
model.add(tf.keras.layers.GlobalAveragePooling3D())

# Dense layer for SOC prediction
model.add(Dense(1, activation='sigmoid'))  # Use 'linear' for non-normalized SOC values

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Print the model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('models/convolstm_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
history = model.fit(
    X_train_conv, y_train_seq,
    validation_data=(X_val_conv, y_val_seq),
    epochs=100,
    batch_size=72,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/training_loss.png")
    plt.show()

plot_training_history(history)
