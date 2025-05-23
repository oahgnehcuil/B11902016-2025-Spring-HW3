import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import os

print("Setting parameters...")
assets = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]  # 不要 SPY
start = "2019-01-01"
end = "2024-04-01"
lookback = 50

print("Downloading data from Yahoo Finance...")
df = pd.DataFrame()
for asset in assets:
    print(f"Downloading {asset}...")
    raw = yf.download(asset, start=start, end=end, auto_adjust=False)
    df[asset] = raw['Adj Close']

print("Download complete.")

print("Preprocessing data...")
returns = df.pct_change().fillna(0)
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)
print("Data preprocessing complete.")

print("Creating NN datasets...")
def create_nn_dataset(data, lookback=50):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i].flatten())
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_nn_dataset(returns_scaled, lookback)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

input_dim = lookback * len(assets)
print("Building NN model...")
model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, verbose=True)

print("Model build complete.")

print("Starting training...")
model.fit(X, y)
print("Training complete.")

print("Saving model...")
if not os.path.exists("./model"):
    os.makedirs("./model")
joblib.dump(model, "nn_portfolio_model.pkl")
print("✅ Model saved as nn_portfolio_model.pkl")