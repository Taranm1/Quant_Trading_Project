import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Step 1: Download data
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
    _, pvalue, _ = coint(log_prices[stock1], log_prices[stock2])
    if pvalue < 0.1:
        cointegrated_pairs.append((stock1, stock2, pvalue))

print(f"Found {len(cointegrated_pairs)} cointegrated pairs.")

# Step 3: Feature engineering and labeling
all_features = []
for stock1, stock2, _ in cointegrated_pairs:
    y = log_prices[stock1]
    X = sm.add_constant(log_prices[stock2])
    model = sm.OLS(y, X).fit()
    hedge_ratio = model.params.iloc[1]

    spread = y - hedge_ratio * log_prices[stock2]
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

    df = pd.DataFrame(index=spread.index)
    df['zscore'] = zscore
    df['zscore_1d'] = zscore.shift(1)
    df['zscore_3d'] = zscore.shift(3)
    df['zscore_5d'] = zscore.shift(5)
    df['spread_std_5'] = spread.rolling(5).std()
    df['spread_mean_5'] = spread.rolling(5).mean()
    df['zscore_change'] = zscore - zscore.shift(1)
    df['spread_pct'] = spread / spread_mean
    df['volatility'] = spread.rolling(10).std()
    df['momentum'] = zscore.rolling(3).mean()
    df['bollinger_upper'] = df['spread_mean_5'] + 2 * df['spread_std_5']
    df['bollinger_lower'] = df['spread_mean_5'] - 2 * df['spread_std_5']
    df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']
    df['zscore_velocity'] = df['zscore'] - df['zscore_1d']
    df['zscore_acceleration'] = df['zscore_velocity'] - df['zscore_velocity'].shift(1)

    # Loosened label to increase positive samples
    future_z = zscore.shift(-5)
    df['label'] = ((zscore.abs() > 0.8) & (future_z.abs() < 0.7)).astype(int)

    df['pair'] = f"{stock1}_{stock2}"
    df = df.dropna()
    all_features.append(df)

# Step 4: Combine all features
full_data = pd.concat(all_features)

# Step 5: Prepare features and labels
X = full_data.drop(columns=['label', 'pair'])
y = full_data['label']

# Step 6: Train/test split (no shuffle to preserve time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 7: Use SMOTE to balance classes
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# Step 8: Scale features
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train XGBoost classifier with tuned hyperparameters
model = XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_bal, y_train_bal)

# Step 10: Predict on test set
y_pred = model.predict(X_test_scaled)
print("Classification report (default threshold 0.5):")
print(classification_report(y_test, y_pred))

# Optional Step 11: Try prediction threshold tuning
y_probs = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.3
y_pred_thresh = (y_probs >= threshold).astype(int)
print(f"Classification report (threshold = {threshold}):")
print(classification_report(y_test, y_pred_thresh))
