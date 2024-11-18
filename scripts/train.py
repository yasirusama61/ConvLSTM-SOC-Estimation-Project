"""
Script for training a ConvLSTM model for State of Charge (SOC) estimation.
This script loads preprocessed battery dataset, defines a ConvLSTM-based hybrid architecture, 
and trains the model to predict SOC values.

Key Features:
- **Data Preparation**: Prepares sequential data for ConvLSTM input format.
- **Model Architecture**: Combines convolutional and LSTM layers to capture spatial and temporal features.
- **Training**: Implements callbacks for early stopping, learning rate reduction, and model checkpointing.
- **Evaluation**: Tracks loss and Mean Absolute Error (MAE) during training.

Requirements:
- TensorFlow/Keras
- NumPy
- Matplotlib
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TIME_STEPS = 20  # Sequence length
FEATURES = 5     # Number of input features
VALIDATION_SPLIT = 0.2
MODEL_PATH = 'models/convlstm_model.h5'

# --- Data Preparation ---
def load_data(file_path, input_columns, target_column):
    """
    Load the dataset and split into features and target.
    
    Args:
        file_path (str): Path to the dataset file.
        input_columns (list): List of feature column names.
        target_column (str): Name of the target column.

    Returns:
        X (numpy array): Feature data.
        y (numpy array): Target data.
    """
    data = np.load(file_path, allow_pickle=True)
    X = data[input_columns]
    y = data[target_column]
    return np.array(X), np.array(y)

def reshape_data(X, y):
    """
    Reshape data for ConvLSTM input format.
    
    Args:
        X (numpy array): Feature data.
        y (numpy array): Target data.

    Returns:
        X_reshaped (numpy array): Reshaped feature data.
        y (numpy array): Target data (unchanged).
    """
    X_reshaped = X.reshape((X.shape[0], TIME_STEPS, 1, 1, FEATURES))
    return X_reshaped, y

# Load dataset
print("Loading data...")
data_path = 'data/training_data.npz'  # Replace with dataset path
input_columns = ['Voltage', 'Current', 'Temperature', 'SOC Rolling', 'Current Rolling']
target_column = 'SOC'

X, y = load_data(data_path, input_columns, target_column)
X, y = reshape_data(X, y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# --- Model Definition ---
def build_convlstm_model():
    """
    Define the ConvLSTM model for SOC estimation.

    Returns:
        model (Sequential): Compiled ConvLSTM model.
    """
    model = Sequential()
    model.add(ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        input_shape=(TIME_STEPS, 1, 1, FEATURES),
        activation='relu',
        return_sequences=False
    ))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))  # Output SOC value
    return model

# Build and compile the model
model = build_convlstm_model()
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])

# --- Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

# --- Training ---
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# --- Plot Training History ---
def plot_training_history(history):
    """
    Plot the training and validation loss and MAE.

    Args:
        history: Keras History object containing training metrics.
    """
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.show()

# Plot training history
plot_training_history(history)

print("Training completed. Best model saved at:", MODEL_PATH)
