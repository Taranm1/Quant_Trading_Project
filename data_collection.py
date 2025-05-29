import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import classification_report

# --- Step 1: Download & prepare data ---

tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "NVDA", "ADBE", "INTC", "CSCO", "ORCL",
    "CRM", "IBM", "QCOM", "TXN", "AMD",
    "AVGO", "MU", "HPQ", "DELL", "ZM"
]

print("Downloading price data...")
data = yf.download(tickers, start='2018-01-01', end='2023-01-01')['Close']
data = data.dropna(how='all')  # drop days where all prices are missing
data = data.ffill().bfill()  # forward/backward fill missing prices
log_prices = np.log(data)

# --- Step 2: Find cointegrated pairs ---

print("Finding cointegrated pairs...")
pairs = list(combinations(log_prices.columns, 2))
cointegrated_pairs = []

for stock1, stock2 in pairs:
    score, pvalue, _ = coint(log_prices[stock1], log_prices[stock2])
    if pvalue < 0.1:
        cointegrated_pairs.append((stock1, stock2, pvalue))

print(f"Found {len(cointegrated_pairs)} cointegrated pairs.")

# --- Step 3 & 4: Hedge ratio, spread, feature engineering, and profit-based labeling ---

all_features = []

def compute_profit_label(spread_series, entry_index, exit_index, threshold=0):
    """
    Compute profit from a trade entered at entry_index and exited at exit_index.
    Positive profit if spread moves toward mean (zero).
    Returns 1 if profitable (profit > threshold), else 0.
    """
    entry_spread = spread_series.iloc[entry_index]
    exit_spread = spread_series.iloc[exit_index]
    # Assume you enter a mean-reversion trade: short if spread>0, long if spread<0
    # Profit if spread moves closer to zero
    profit = abs(entry_spread) - abs(exit_spread)
    return int(profit > threshold)

print("Engineering features and labeling by actual profit for each pair...")

window_exit = 5  # days to hold trade and check profit

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

    # Label: 1 if entering trade at day t is profitable by day t+window_exit
    labels = []
    for i in range(len(df) - window_exit):
        z = df['zscore'].iloc[i]
        # Only consider signals when spread is large enough to enter trade
        if abs(z) > 1.5:
            label = compute_profit_label(spread, i, i + window_exit, threshold=0)
        else:
            label = 0
        labels.append(label)
    # Pad the last window_exit days with 0 (no label)
    labels.extend([0]*window_exit)
    df['label'] = labels

    df['pair'] = f"{stock1}_{stock2}"
    df = df.dropna()
    all_features.append(df)

# --- Step 5: Combine data and prepare for ML training ---

full_data = pd.concat(all_features)
print(f"Total samples after feature engineering: {len(full_data)}")

X = full_data.drop(columns=['label', 'pair'])
y = full_data['label']

# --- Step 6: Expanding window training + evaluation ---

initial_train_ratio = 0.7
test_ratio = 0.1
step_ratio = 0.1

n_samples = len(full_data)
initial_train_size = int(n_samples * initial_train_ratio)
test_size = int(n_samples * test_ratio)
step_size = int(n_samples * step_ratio)

all_reports = []

print(f"Training sizes: initial train {initial_train_size}, test {test_size}, step {step_size}")

for start in range(initial_train_size, n_samples - test_size + 1, step_size):
    train_idx = range(0, start)
    test_idx = range(start, start + test_size)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

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

# --- Step 7: Simple backtest based on ML predictions ---

print("Running simple backtest on last test window...")

# Using last trained model and last test set to backtest profits
test_signals = y_pred  # model predicted labels
test_data = full_data.iloc[test_idx].copy()
test_data['pred_signal'] = test_signals

# Backtest params
transaction_cost = 0.0005  # 5 bps per trade
holding_period = window_exit

# Function to compute PnL per trade signal in test data
def backtest_pnl(df):
    pnl = 0
    trades = 0
    wins = 0
    for i in range(len(df) - holding_period):
        if df['pred_signal'].iloc[i] == 1:
            trades += 1
            spread_entry = df['zscore'].iloc[i]
            spread_exit = df['zscore'].iloc[i + holding_period]
            profit = abs(spread_entry) - abs(spread_exit)  # profit if spread moves toward zero
            profit -= 2 * transaction_cost  # entry + exit costs
            pnl += profit
            if profit > 0:
                wins += 1
    return pnl, trades, wins

pnl, trades, wins = backtest_pnl(test_data)
print(f"Backtest results on last test set:")
print(f"Trades executed: {trades}")
print(f"Winning trades: {wins}")
print(f"Total PnL (z-score units): {pnl:.4f}")
if trades > 0:
    print(f"Win rate: {wins/trades:.2%}")
    print(f"Average PnL per trade: {pnl/trades:.4f}")
else:
    print("No trades executed.")

print("All done.")
