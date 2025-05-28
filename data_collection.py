import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", 
    "NVDA", "ADBE", "INTC", "CSCO", "ORCL",
    "CRM", "IBM", "QCOM", "TXN", "AMD", 
    "AVGO", "MU", "HPQ", "DELL", "ZM"
]
data=yf.download(tickers, start='2018-01-01', end='2023-01-01')

# 3. Extract daily closing prices
close_prices = data['Close']

# 4. Handle missing values (if any)
close_prices = close_prices.dropna()

# 5. Visualize the closing prices
close_prices.plot(figsize=(12,6), title='Daily Close Prices')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)



log_prices=np.log(close_prices)
pairs=list(combinations(log_prices.columns, 2))



cointegrated_pairs = []

for stock1, stock2 in pairs:
    score, pvalue, _ = coint(log_prices[stock1], log_prices[stock2])
    print(f"Testing pair: {stock1} & {stock2} | p-value: {pvalue:.4f}")
    if pvalue < 0.05:  # statistically significant
        cointegrated_pairs.append((stock1, stock2, pvalue))

print(f"Found {len(cointegrated_pairs)} cointegrated pairs.")


# Sort by p-value (strongest cointegration first)
cointegrated_pairs.sort(key=lambda x: x[2])

# Print result
for pair in cointegrated_pairs:
    print(f"Pair: {pair[0]} & {pair[1]} | p-value: {pair[2]:.4f}")

plt.show()

for stock1, stock2, pvalue in cointegrated_pairs:
    y = log_prices[stock1]                  # Series of log prices for stock1
    X = sm.add_constant(log_prices[stock2])  # Series of log prices for stock2, with constant added

    model = sm.OLS(y, X).fit()              # Run linear regression with y and X (actual price data)
    hedge_ratio = model.params[1]           # slope is the hedge ratio

    spread = y - hedge_ratio * log_prices[stock2]  # Calculate spread using actual price series

    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

entry_threshold = 2
exit_threshold = 0.5

# Initialize a DataFrame to store signals
signals = pd.DataFrame(index=spread.index)
signals['zscore'] = zscore

# Long entry signal: spread z-score < -entry_threshold
signals['long_entry'] = signals['zscore'] < -entry_threshold

# Short entry signal: spread z-score > entry_threshold
signals['short_entry'] = signals['zscore'] > entry_threshold

# Exit signal: when z-score between -exit_threshold and +exit_threshold
signals['exit'] = signals['zscore'].abs() < exit_threshold

# Position tracking:
# +1 for long, -1 for short, 0 for no position

signals['position'] = 0

for i in range(1, len(signals)):
    if signals['long_entry'].iloc[i]:
        signals.at[signals.index[i], 'position'] = 1
    elif signals['short_entry'].iloc[i]:
        signals.at[signals.index[i], 'position'] = -1
    elif signals['exit'].iloc[i]:
        signals.at[signals.index[i], 'position'] = 0
    else:
        signals.at[signals.index[i], 'position'] = signals['position'].iloc[i-1]



features = pd.DataFrame(index=spread.index)
features['zscore'] = zscore
features['zscore_1d'] = zscore.shift(1)
features['zscore_3d'] = zscore.shift(3)
features['zscore_5d'] = zscore.shift(5)
features['spread_std_5'] = spread.rolling(5).std()
features['spread_mean_5'] = spread.rolling(5).mean()
features = features.dropna()

# Reversion = z-score crossing back toward 0 from an extreme
future_zscore = zscore.shift(-3)  # 3-day future z-score
labels = (zscore.abs() > 2) & (future_zscore.abs() < 1)
features['label'] = labels.astype(int)

from sklearn.model_selection import train_test_split

X = features.drop(columns=['label'])
y = features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))