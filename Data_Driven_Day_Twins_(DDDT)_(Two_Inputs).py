# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:42:07 2024

@author: W10
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GRU, Dense, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tqdm import tqdm

# Normalize data
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Denormalize data
def denormalize_data(normalized_data, original_min, original_max):
    return normalized_data * (original_max - original_min) + original_min

# Load data from Excel
file_path = r"E:\Belgeler\Calismalar\Benzerlik Tahmin\Veri3.xlsx"

# Load training data
input1 = pd.read_excel(file_path, sheet_name="Input1").values.flatten()
input2 = pd.read_excel(file_path, sheet_name="Input3").values.flatten()
output = pd.read_excel(file_path, sheet_name="Output").values.flatten()

# Load test data
input_test1 = pd.read_excel(file_path, sheet_name="Input_Test1").values.flatten()
input_test2 = pd.read_excel(file_path, sheet_name="Input_Test3").values.flatten()
output_test = pd.read_excel(file_path, sheet_name="Output_Test").values.flatten()

# Save min and max for denormalization
original_output_min, original_output_max = np.min(output), np.max(output)

# Normalize data
input1 = normalize_data(input1)
input2 = normalize_data(input2)
output = normalize_data(output)

input_test1 = normalize_data(input_test1)
input_test2 = normalize_data(input_test2)
output_test = normalize_data(output_test)

# Create sequences for training
def create_sequences(input_data1, input_data2, output_data, window_size=24):
    X, y = [], []
    for i in range(len(output_data) - window_size):
        X.append(np.stack([input_data1[i:i+window_size],
                          input_data2[i:i+window_size]], axis=-1))
        y.append(output_data[i+window_size])
    return np.array(X), np.array(y)

tf.keras.utils.set_random_seed(42)  # Tüm random operasyonları sabitler
# Prepare training data
X_train, y_train = create_sequences(input1, input2, output)
X_train = X_train[..., np.newaxis]  # Add channel dimension

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

# Build model with fixed parameters
def build_model():
    input_layer = Input(shape=(24, 2, 1), name='input')  # Changed to 2 input features

    # Fixed hyperparameters
    filters = 48
    kernel_size = (8, 2)  # Height, Width

    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(input_layer)
    x = Reshape((24, 2 * filters))(x)  # Changed to 2 input features

    gru_units = 400
    x = GRU(gru_units)(x)

    output_layer = Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(scheduler)

# Build and train model
model = build_model()
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler],
    shuffle=False
)

# Modified prediction function
def iterative_predict(model, input_test1, input_test2, output_test, window_size=24):
    predictions = []
    current_window = np.stack([input_test1[:window_size],
                              input_test2[:window_size]], axis=-1)  # Removed third input
    current_window = current_window.astype(np.float32)[np.newaxis, ..., np.newaxis]

    max_steps = min(len(output_test), len(input_test1) - window_size)

    for i in tqdm(range(max_steps), desc="Predicting"):
        if current_window.shape != (1, 24, 2, 1):  # Adjusted for 2 features
            current_window = current_window.reshape(1, 24, 2, 1)

        pred = model.predict_on_batch(current_window)
        pred_value = pred.flat[0]
        predictions.append(pred_value)

        if i < max_steps - 1:
            new_idx = window_size + i
            new_input1 = input_test1[new_idx]
            new_input2 = input_test2[new_idx]

            current_window = np.roll(current_window, shift=-1, axis=1)
            current_window[0, -1, :, 0] = [new_input1, new_input2]  # Removed third input

    return np.array(predictions)

# Make predictions
predictions_test = iterative_predict(model, input_test1, input_test2, output_test)

# Denormalize and calculate metrics
predictions_test_denorm = denormalize_data(predictions_test, original_output_min, original_output_max)
output_test_denorm = denormalize_data(output_test[:len(predictions_test)], original_output_min, original_output_max)

rmse_test_denorm = np.sqrt(mean_squared_error(output_test_denorm, predictions_test_denorm))
mae_test_denorm = mean_absolute_error(output_test_denorm, predictions_test_denorm)
r2_test_denorm = r2_score(output_test_denorm, predictions_test_denorm)
mape_test_denorm = mean_absolute_percentage_error(output_test_denorm, predictions_test_denorm)

print(f'Test RMSE (Denormalize): {rmse_test_denorm}')
print(f'Test MAE (Denormalize): {mae_test_denorm}')
print(f'Test R² (Denormalize): {r2_test_denorm}')
print(f'Test MAPE (Denormalize): {mape_test_denorm}')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(output_test_denorm, label='Actual Values')
plt.plot(predictions_test_denorm, label='Predicted Values')
plt.title('Actual vs Predicted Values (Test Data)')
plt.xlabel('Hour')
plt.ylabel('Value')
plt.legend()
plt.show()