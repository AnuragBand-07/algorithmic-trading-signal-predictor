"""Feature engineering.

Raw prices are non-stationary and leak trend information, so the model is fed
momentum / trend *oscillators* and ratio features instead of price levels.

All indicators use only information available up to and including day ``t``.
The target uses day ``t+1`` and is therefore the only forward-looking column;
its last row is NaN and is dropped, preventing look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns actually fed to the model. >5 features, all stationary/bounded.
FEATURE_COLUMNS = [
    "return_1d",      # yesterday's return (momentum)
    "return_5d",      # 1-week momentum
    "close_sma10",    # Close / SMA(10) - 1  (short-term trend)
    "close_sma50",    # Close / SMA(50) - 1  (long-term trend)
    "sma10_sma50",    # SMA(10) / SMA(50) - 1 (trend crossover)
    "ema12_ema26",    # EMA(12) / EMA(26) - 1 (MACD basis, normalised)
    "rsi_14",         # Relative Strength Index (0-100 momentum oscillator)
    "macd_hist",      # MACD histogram (momentum acceleration)
    "volatility_10",  # rolling std of returns (risk regime)
    "volume_ratio",   # Volume / 20d avg volume (participation)
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI (uses exponential smoothing, not a simple average)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing == EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0)  # zero losses => RSI 100


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with engineered features + binary target ``y``.

    ``y = 1`` if the next day's close is higher than today's, else ``0``.
    """
    df = prices.copy()
    close = df["Close"]

    # --- Moving averages (SMA / EMA) ---------------------------------------
    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    # --- Momentum / returns ------------------------------------------------
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)

    # --- Trend features (normalised so they are stationary) ----------------
    df["close_sma10"] = close / sma10 - 1
    df["close_sma50"] = close / sma50 - 1
    df["sma10_sma50"] = sma10 / sma50 - 1
    df["ema12_ema26"] = ema12 / ema26 - 1

    # --- Oscillators -------------------------------------------------------
    df["rsi_14"] = _rsi(close, 14)
    _, _, macd_hist = _macd(close)
    df["macd_hist"] = macd_hist / close  # normalise by price level

    # --- Risk / participation regime ---------------------------------------
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- Target: next-day direction (the ONLY forward-looking column) ------
    df["future_return"] = close.pct_change().shift(-1)
    df["y"] = (df["future_return"] > 0).astype(int)

    # Drop warm-up NaNs (from rolling windows) and the final NaN target row.
    df = df.dropna(subset=FEATURE_COLUMNS + ["future_return"])
    return df


if __name__ == "__main__":
    from .data import load_prices

    feats = build_features(load_prices())
    print(feats[FEATURE_COLUMNS + ["y"]].tail())
    print("\nTarget balance (base rate):")
    print(feats["y"].value_counts(normalize=True).round(3))
