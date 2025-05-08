import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
import os

# 1. Parameters
print("Setting parameters...")
assets = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]  # 不要 SPY
start = "2019-01-01"
end = "2024-01-01"
lookback = 50   # LSTM看過去多少天

# 2. Download Data
print("Downloading data from Yahoo Finance...")
df = pd.DataFrame()
for asset in assets:
    print(f"Downloading {asset}...")
    raw = yf.download(asset, start=start, end=end, auto_adjust=False)
    df[asset] = raw['Adj Close']

print("Download complete.")

# 3. Preprocess Data
print("Preprocessing data...")
returns = df.pct_change().fillna(0)
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)
print("Data preprocessing complete.")

# 4. Create dataset for LSTM
print("Creating LSTM datasets...")
def create_dataset(data, lookback=50):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_dataset(returns_scaled, lookback)

print(f"X shape: {X.shape}")  # (樣本數, 50, 資產數)
print(f"y shape: {y.shape}")  # (樣本數, 資產數)

# 5. Build LSTM Model
print("Building LSTM model...")
model = Sequential()
model.add(LSTM(64, input_shape=(lookback, len(assets)), return_sequences=False))
model.add(Dense(len(assets), activation='linear'))

model.compile(optimizer='adam', loss=MeanSquaredError())
print("Model build complete.")

# 6. Train Model
print("Starting training...")
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)
print("Training complete.")

# 7. Save Model
print("Saving model...")
if not os.path.exists("./model"):
    os.makedirs("./model")
model.save("lstm_portfolio_model.h5")
print("✅ Model saved as lstm_portfolio_model.h5")
