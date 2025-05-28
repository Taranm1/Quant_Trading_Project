import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from collections import Counter

# Step 1: Download data for given tickers
tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "NVDA", "ADBE", "INTC", "CSCO", "ORCL",
    "CRM", "IBM", "QCOM", "TXN", "AMD",
    "AVGO", "MU", "HPQ", "DELL", "ZM"
]

print("Downloading price data...")
data = yf.download(tickers, start='2018-01-01', end='2023-01-01')
close_prices = data['Close'].dropna()
log_prices = np.log(close_prices)

# Step 2: Find cointegrated pairs (p-value < 0.1)
print("Finding cointegrated pairs...")
pairs = list(combinations(log_prices.columns, 2))
cointegrated_pairs = []

for stock1, stock2 in pairs:
    score, pvalue, _ = coint(log_prices[stock1], log_prices[stock2])
    if pvalue < 0.1:
        cointegrated_pairs.append((stock1, stock2, pvalue))

print(f"Found {len(cointegrated_pairs)} cointegrated pairs.")

# Step 3: Feature engineering and labeling for each pair
all_features = []

print("Engineering features for each pair...")
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

    # Label: loosened to increase positives
    future_z = zscore.shift(-5)
    df['label'] = ((zscore.abs() > 0.8) & (future_z.abs() < 0.7)).astype(int)

    df['pair'] = f"{stock1}_{stock2}"
    df = df.dropna()
    all_features.append(df)

# Step 4: Combine all pairs data
full_data = pd.concat(all_features)
print(f"Total samples: {len(full_data)}")

# Step 5: Prepare features and labels
X = full_data.drop(columns=['label', 'pair'])
y = full_data['label']

# Parameters for expanding window training
initial_train_ratio = 0.7
test_ratio = 0.1
step_ratio = 0.1

n_samples = len(full_data)
initial_train_size = int(n_samples * initial_train_ratio)
test_size = int(n_samples * test_ratio)
step_size = int(n_samples * step_ratio)

print(f"Initial train size: {initial_train_size}, test size: {test_size}, step size: {step_size}")

all_reports = []

for start in range(initial_train_size, n_samples - test_size + 1, step_size):
    train_idx = range(0, start)
    test_idx = range(start, start + test_size)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Check class distribution in training set before applying SMOTE
    counter = Counter(y_train)
    if len(counter) < 2:
        print(f"Skipping SMOTE at train size {len(y_train)} due to single class: {counter}")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=300,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        print(f"Applying SMOTE at train size {len(y_train)}: {counter}")
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_bal = scaler.fit_transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)

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
        y_pred = model.predict(X_test_scaled)

    print(f"Classification report for test period starting at index {start}:")
    print(classification_report(y_test, y_pred))
    all_reports.append(classification_report(y_test, y_pred, output_dict=True))

print("Done all splits.")
