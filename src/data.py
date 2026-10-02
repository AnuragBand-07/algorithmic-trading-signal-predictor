"""Data acquisition layer.

Fetches historical OHLCV data via yfinance and caches it locally as parquet so
the rest of the pipeline is reproducible and does not hammer Yahoo's (flaky,
rate-limited, unofficial) API on every run.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import config


def _cache_path(ticker: str, period: str, interval: str):
    safe = ticker.replace("^", "IDX_").replace(".", "_")
    return config.DATA_DIR / f"{safe}_{period}_{interval}.parquet"


def load_prices(
    ticker: str = config.TICKER,
    period: str = config.PERIOD,
    interval: str = config.INTERVAL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a clean OHLCV frame indexed by date.

    Columns: Open, High, Low, Close, Volume (all float).
    Uses a local parquet cache unless ``force_refresh`` is True.
    """
    path = _cache_path(ticker, period, interval)

    if path.exists() and not force_refresh:
        df = pd.read_parquet(path)
    else:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,   # adjust for splits/dividends
            progress=False,
        )
        if df.empty:
            raise ValueError(
                f"No data returned for '{ticker}'. Check the ticker symbol "
                f"(NIFTY 50 = '^NSEI', Reliance = 'RELIANCE.NS')."
            )
        # yfinance can return a MultiIndex (Price, Ticker) for single tickers.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.to_parquet(path)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna(subset=["Close"])
    return df


if __name__ == "__main__":
    prices = load_prices(force_refresh=True)
    print(f"Loaded {len(prices)} rows for {config.TICKER}")
    print(prices.tail())
