import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. SETTINGS ---
n_past = 7      # Number of past days to use as input
n_future = 2    # Number of future days to predict

# --- 2. HELPER FUNCTION ---
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []

    # Input sequence (t-n ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'var{j+1}(t-{i})') for j in range(n_vars)]

    # Forecast sequence (t, t+1 ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        names += [(f'var{j+1}(t+{i})') for j in range(n_vars)]

    # Combine all
    agg = concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg

# --- 3. LOAD AND PREPROCESS DATA ---
file_path = r"technical_indicators\BHP\Delta\BHPdelta.csv"
dataset = pd.read_csv(file_path, header=0, index_col=0)
values = dataset.values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# Supervised learning format
reframed = series_to_supervised(scaled, n_past, n_future)
print(reframed.head())
print("Reframed shape:", reframed.shape)

# --- 4. SPLIT INTO TRAIN/TEST ---
values = reframed.values
train = values[:1079, :]
test = values[1080:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print(train_X.shape, test_X.shape)

# --- 5. DEFINE KERAS MLP MODEL ---
def create_mlp_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    return model

model = create_mlp_model(train_X.shape[1])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# --- 6. TRAIN THE MODEL ---
history = model.fit(
    train_X,
    train_y,
    validation_data=(test_X, test_y),
    epochs=450,
    batch_size=64,
    verbose=2
)

# --- 7. EVALUATE MODEL ---
y_pred = model.predict(test_X)

rmse = np.sqrt(mean_squared_error(test_y, y_pred))
mse = mean_squared_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)
print(f"RMSE: {rmse:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

# --- 8. PLOT LOSS ---
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# --- 9. SAVE 19 PREDICTIONS TO EXCEL ---
import openpyxl

# Select first 19 predictions and actual values
num_values = 19
predicted_values = y_pred[:num_values].flatten()
actual_values = test_y[:num_values]

# Create DataFrame
results_df = pd.DataFrame({
    'Predicted': predicted_values,
    'Actual': actual_values
})

# Save to Excel
output_excel_path = "BHPDELTA.xlsx"
results_df.to_excel(output_excel_path, index=False)

print(f"Saved {num_values} predicted vs actual values to '{output_excel_path}'")
