import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 多家科技股代碼
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "ORCL", "INTC", "AMD", "QUBT", "ASML", "TSM", "SMCI", "ARM", "JBLU", "QQQ"]
print("Tickers:", tickers)

# 抓 Close 價格（注意 auto_adjust=False，這樣才有原始 Close 欄位）
data = yf.download(tickers, start="2022-01-01", end="2024-01-01", auto_adjust=False)

# 拿出 Close 欄位（是 MultiIndex）
if isinstance(data.columns, pd.MultiIndex):
    data = data['Close']  # 只取出 Close 層
print("Data shape:", data.shape)
print(data.head())

# 計算每日報酬率
returns = data.pct_change().dropna()
print("Returns shape:", returns.shape)

# 計算相關係數矩陣
correlation_matrix = returns.corr()

# 畫相關係數熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Major Tech Stocks (2022 Jan – 2024 Jan)")
plt.tight_layout()
plt.show()

# Assume `returns` is a DataFrame: rows are time, columns are tickers
correlation_matrix = returns.corr()

# Build edge list for correlation > 0.5 (excluding self-loops)
edges = []
threshold = 0.5
for i, stock1 in enumerate(correlation_matrix.columns):
    for j, stock2 in enumerate(correlation_matrix.columns):
        if i < j and correlation_matrix.iloc[i, j] > threshold:
            edges.append((i, j))
