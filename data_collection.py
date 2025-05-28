import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Download Data
tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", 
    "NVDA", "ADBE", "INTC", "CSCO", "ORCL",
    "CRM", "IBM", "QCOM", "TXN", "AMD", 
    "AVGO", "MU", "HPQ", "DELL", "ZM"
]

data = yf.download(tickers, start='2018-01-01', end='2023-01-01')
close_prices = data['Close'].dropna()
log_prices = np.log(close_prices)

# Step 2: Find cointegrated pairs
pairs = list(combinations(log_prices.columns, 2))
cointegrated_pairs = []

for stock1, stock2 in pairs:
    score, pvalue, _ = coint(log_prices[stock1], log_prices[stock2])
    if pvalue < 0.05:
        cointegrated_pairs.append((stock1, stock2, pvalue))

cointegrated_pairs.sort(key=lambda x: x[2])
print(f"Found {len(cointegrated_pairs)} cointegrated pairs.")

# Step 3: Extract features and labels from all cointegrated pairs
all_features = []

for stock1, stock2, pvalue in cointegrated_pairs:
    y = log_prices[stock1]
    X = sm.add_constant(log_prices[stock2])
    model = sm.OLS(y, X).fit()
    hedge_ratio = model.params.iloc[1]

    spread = y - hedge_ratio * log_prices[stock2]
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

    # Build signal features
    df = pd.DataFrame(index=spread.index)
    df['zscore'] = zscore
    df['zscore_1d'] = zscore.shift(1)
    df['zscore_3d'] = zscore.shift(3)
    df['zscore_5d'] = zscore.shift(5)
    df['spread_std_5'] = spread.rolling(5).std()
    df['spread_mean_5'] = spread.rolling(5).mean()
    
    # Labels: mean reversion signal
    future_zscore = zscore.shift(-7)
    df['label'] = ((zscore.abs() > 1) & (future_zscore.abs() < 1)).astype(int)
    
    # Add pair info (optional, helpful for analysis)
    df['pair'] = f"{stock1}_{stock2}"
    
    # Drop missing values
    df = df.dropna()

    all_features.append(df)

# Step 4: Combine features from all pairs
full_data = pd.concat(all_features)

# Step 5: Split into train/test and train the model
X = full_data.drop(columns=['label', 'pair'])
y = full_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
