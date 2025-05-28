import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

tickers = [
    "AAPL", "ADBE", "AMD", "AMZN", "AVGO", "CRM", "CSCO", "DELL",
    "GOOG", "HPQ", "IBM", "INTC", "META", "MSFT", "MU", "NVDA",
    "ORCL", "QCOM", "TXN", "ZM"
]

print("Downloading data for all tickers at once...")
data = yf.download(tickers, start="2020-01-01", end="2024-12-31")

print(f"Columns in downloaded data:\n{data.columns}")

# Function to get adjusted close prices or fallback
def get_adj_close(df, tickers):
    # If multi-level columns (multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            print("Using 'Adj Close' from multi-index columns.")
            return df['Adj Close']
        elif 'Close' in df.columns.levels[0]:
            print("Warning: 'Adj Close' not found, using 'Close' from multi-index columns.")
            return df['Close']
        else:
            raise KeyError("No 'Adj Close' or 'Close' in multi-index columns.")
    else:
        # Single ticker or single level columns
        if 'Adj Close' in df.columns:
            print("Using 'Adj Close' from columns.")
            return df['Adj Close']
        elif 'Close' in df.columns:
            print("Warning: 'Adj Close' not found, using 'Close' from columns.")
            return df['Close']
        else:
            # If nothing found, download individually per ticker
            print("No 'Adj Close' or 'Close' found in single DataFrame, downloading individually.")
            all_adj = []
            for t in tickers:
                print(f"Downloading data for {t} individually...")
                single_df = yf.download(t, start="2020-01-01", end="2024-12-31")
                if 'Adj Close' in single_df.columns:
                    all_adj.append(single_df['Adj Close'].rename(t))
                elif 'Close' in single_df.columns:
                    print(f"Warning: {t} no 'Adj Close', using 'Close' instead.")
                    all_adj.append(single_df['Close'].rename(t))
                else:
                    raise KeyError(f"No 'Adj Close' or 'Close' for ticker {t}.")
            adj_close_df = pd.concat(all_adj, axis=1)
            return adj_close_df

adj_close = get_adj_close(data, tickers)

adj_close.dropna(inplace=True)
print(f"Data shape after cleaning: {adj_close.shape}")

print("Testing pairs for cointegration...")
n = adj_close.shape[1]
pvalue_matrix = np.ones((n, n))
keys = adj_close.columns
pairs = []

for i in range(n):
    for j in range(i + 1, n):
        S1 = adj_close[keys[i]]
        S2 = adj_close[keys[j]]
        result = coint(S1, S2)
        pvalue = result[1]
        pvalue_matrix[i, j] = pvalue
        print(f"Testing pair: {keys[i]} & {keys[j]} | p-value: {pvalue:.4f}")
        if pvalue < 0.1:  # 10% significance level
            pairs.append((keys[i], keys[j]))

print(f"\nFound {len(pairs)} cointegrated pairs (p < 0.1).")

def create_features_and_target(S1, S2, window=5, threshold=0.5):
    spread = S1 - S2
    spread_ma = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    zscore = (spread - spread_ma) / spread_std

    df = pd.DataFrame({
        'zscore': zscore,
        'zscore_lag1': zscore.shift(1),
        'zscore_lag2': zscore.shift(2),
        'zscore_lag3': zscore.shift(3),
    }).dropna()

    df['target'] = (zscore.shift(-1) < -threshold).astype(int)

    return df.drop('target', axis=1), df['target']

for s1, s2 in pairs:
    print(f"\n--- Processing pair: {s1} & {s2} ---")

    X, y = create_features_and_target(adj_close[s1], adj_close[s2])

    if y.nunique() < 2:
        print(f"Only one class present in target for pair {s1} & {s2}. Skipping ML training.")
        continue

    if y.sum() == 0:
        print(f"No positive labels for pair {s1} & {s2}, skipping ML training.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    try:
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        print(f"SMOTE error for pair {s1} & {s2}: {e}. Skipping this pair.")
        continue

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    print(f"Classification report for pair {s1} & {s2}:\n")
    print(classification_report(y_test, y_pred))

print("\nAll pairs processed.")
