"""Stock data loading, feature engineering, and preprocessing.

WHAT HAPPENS HERE:
1. Download historical stock prices from Yahoo Finance
2. Calculate technical indicators (features the model learns from)
3. Normalize all features to 0-1 range (critical for neural networks)
4. Create sliding window sequences for LSTM input
5. Split into train/test sets

WHY TECHNICAL INDICATORS?
Raw price alone isn't enough — the model needs to understand patterns:
- Moving Averages: Is the price trending up or down?
- RSI: Is the stock overbought (>70) or oversold (<30)?
- MACD: Is momentum building or fading?
- Bollinger Bands: Is the price near its statistical extremes?
- Volume: Is there conviction behind price moves?

WHY SEQUENCES?
LSTMs process data step-by-step through time. We feed 60 days of features
and ask "what happens on day 61?" The model learns temporal patterns like:
- "When RSI drops below 30 and volume spikes, price tends to bounce"
- "When price crosses above SMA_50, it often continues up"

SEQUENCE FORMAT:
  Input: [day1, day2, ..., day60] where each day has 12 features
  Shape: [batch_size, 60, 12]
  Target: closing price on day 61 (or day 61-65 for multi-day forecast)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch


class StockDataPipeline:
    """Downloads, engineers features, and prepares stock data for LSTM."""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()  # Separate scaler for inverse transform
        self.feature_names = config["data"]["features"]
        self.seq_length = config["data"]["sequence_length"]
        self.forecast_days = config["data"]["forecast_days"]

    def fetch_data(self, ticker: str = None) -> pd.DataFrame:
        """Download historical stock data from Yahoo Finance.

        Args:
            ticker: Stock symbol (e.g., 'AAPL'). Defaults to config.

        Returns:
            DataFrame with OHLCV data.
        """
        ticker = ticker or self.config["data"]["ticker"]
        period = self.config["data"]["period"]

        print(f"Fetching {ticker} data ({period})...")
        df = yf.download(ticker, period=period, progress=False)

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        print(f"  Downloaded {len(df)} trading days")
        print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from raw OHLCV data.

        Each indicator captures a different aspect of price behavior:

        TREND INDICATORS (where is the price going?):
        - SMA (Simple Moving Average): Average of last N days
        - EMA (Exponential Moving Average): Recent prices weighted more heavily

        MOMENTUM INDICATORS (how fast is it moving?):
        - RSI (Relative Strength Index): 0-100, measures speed of price changes
        - MACD: Difference between fast and slow EMAs

        VOLATILITY INDICATORS (how much is it swinging?):
        - Bollinger Bands: Price channels based on standard deviation
        - Volatility: Rolling standard deviation of returns
        """
        data = df.copy()

        # === Trend Indicators ===
        # SMA: Simple average of closing prices over a window
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["SMA_50"] = data["Close"].rolling(window=50).mean()

        # EMA: Exponential moving average (reacts faster to recent prices)
        data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

        # === Momentum Indicators ===
        # RSI: Measures if stock is overbought (>70) or oversold (<30)
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        # MACD: When fast EMA crosses above slow EMA = bullish signal
        data["MACD"] = data["EMA_12"] - data["EMA_26"]

        # === Volatility Indicators ===
        # Bollinger Bands: Price channels (mean +/- 2 standard deviations)
        bb_sma = data["Close"].rolling(window=20).mean()
        bb_std = data["Close"].rolling(window=20).std()
        data["BB_upper"] = bb_sma + (bb_std * 2)
        data["BB_lower"] = bb_sma - (bb_std * 2)

        # Daily return: Percentage change from yesterday
        data["Daily_Return"] = data["Close"].pct_change()

        # Volatility: How much the price swings (20-day rolling std of returns)
        data["Volatility"] = data["Daily_Return"].rolling(window=20).std()

        # Drop NaN rows (from rolling calculations)
        data = data.dropna()

        print(f"  Engineered {len(self.feature_names)} features, {len(data)} samples remaining")
        return data

    def prepare_data(self, ticker: str = None):
        """Full pipeline: fetch -> engineer -> normalize -> sequence -> split.

        Returns:
            Dictionary with train/test DataLoaders, scalers, and metadata.
        """
        # Fetch and engineer
        raw_df = self.fetch_data(ticker)
        df = self.engineer_features(raw_df)

        # Extract features
        feature_data = df[self.feature_names].values
        dates = df.index

        # Fit price scaler on Close prices (for inverse transform later)
        close_idx = self.feature_names.index("Close")
        self.price_scaler.fit(feature_data[:, close_idx].reshape(-1, 1))

        # Normalize all features to [0, 1]
        # Neural networks work best when all inputs are on the same scale
        # Without this, features with large values (price=150) would dominate
        # over features with small values (RSI=0.5)
        scaled_data = self.scaler.fit_transform(feature_data)

        # Create sequences
        X, y, seq_dates = self._create_sequences(scaled_data, dates)

        # Train/test split (time-based, NOT random!)
        # We MUST split by time for financial data — you can't train on
        # future data and predict the past (that's "look-ahead bias")
        split_idx = int(len(X) * self.config["data"]["train_ratio"])

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train = seq_dates[:split_idx]
        dates_test = seq_dates[split_idx:]

        print(f"  Train: {len(X_train)} sequences")
        print(f"  Test:  {len(X_test)} sequences")

        # Create DataLoaders
        batch_size = self.config["training"]["batch_size"]
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "dates_train": dates_train,
            "dates_test": dates_test,
            "df": df,
            "raw_df": raw_df,
        }

    def _create_sequences(self, data: np.ndarray, dates: pd.DatetimeIndex):
        """Create sliding window sequences for LSTM input.

        Example with seq_length=3, forecast_days=1:
          Data: [d1, d2, d3, d4, d5, d6]
          Seq 1: X=[d1,d2,d3] → y=d4 (close price)
          Seq 2: X=[d2,d3,d4] → y=d5
          Seq 3: X=[d3,d4,d5] → y=d6
        """
        X, y = [], []
        seq_dates = []
        close_idx = self.feature_names.index("Close")

        for i in range(self.seq_length, len(data) - self.forecast_days + 1):
            # Input: seq_length days of all features
            X.append(data[i - self.seq_length : i])
            # Target: closing price forecast_days ahead
            y.append(data[i + self.forecast_days - 1, close_idx])
            seq_dates.append(dates[i + self.forecast_days - 1])

        return np.array(X), np.array(y), seq_dates

    def inverse_transform_price(self, scaled_prices: np.ndarray) -> np.ndarray:
        """Convert normalized prices back to dollar values."""
        return self.price_scaler.inverse_transform(
            scaled_prices.reshape(-1, 1)
        ).flatten()


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
