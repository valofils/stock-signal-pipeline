"""
Data ingestion module.
Fetches historical OHLCV data for a basket of stocks using yfinance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# 15 well-known S&P 500 tickers across different sectors
TICKERS = [
    "AAPL", "MSFT", "GOOGL",   # Technology
    "JPM", "BAC", "GS",        # Finance
    "JNJ", "PFE", "UNH",       # Healthcare
    "XOM", "CVX",              # Energy
    "AMZN", "WMT",             # Consumer
    "CAT", "BA",               # Industrials
]

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_historical(
    tickers: list[str] = TICKERS,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical daily OHLCV data for a list of tickers.

    Args:
        tickers: List of stock ticker symbols.
        period: Lookback period (e.g. '2y', '1y', '6mo').
        interval: Data frequency (e.g. '1d', '1wk').

    Returns:
        DataFrame with MultiIndex columns (field, ticker).
    """
    print(f"Fetching data for {len(tickers)} tickers | period={period} | interval={interval}")
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=True,
    )
    print(f"Downloaded shape: {df.shape}")
    return df


def save_raw(df: pd.DataFrame, filename: str = "ohlcv_raw.parquet") -> Path:
    """
    Save raw data to the data/raw directory as parquet.

    Args:
        df: Raw OHLCV DataFrame.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_parquet(path)
    print(f"Saved raw data to {path}")
    return path


def load_raw(filename: str = "ohlcv_raw.parquet") -> pd.DataFrame:
    """
    Load raw data from the data/raw directory.

    Args:
        filename: File to load.

    Returns:
        Raw OHLCV DataFrame.
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No raw data found at {path}. Run fetch_historical() first.")
    df = pd.read_parquet(path)
    print(f"Loaded raw data from {path} | shape: {df.shape}")
    return df


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape wide MultiIndex DataFrame into a long format with columns:
    [Date, Ticker, Open, High, Low, Close, Volume]

    Args:
        df: Raw wide-format DataFrame from yfinance.

    Returns:
        Long-format DataFrame.
    """
    df = df.stack(level=1).reset_index()
    df.columns.name = None
    df = df.rename(columns={"Ticker": "Ticker", "Price": "Ticker"})

    # Ensure clean column names
    df = df.rename(columns={df.columns[1]: "Ticker"})
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"Long format shape: {df.shape}")
    return df


if __name__ == "__main__":
    raw = fetch_historical()
    save_raw(raw)
    long_df = to_long_format(raw)
    print(long_df.head(10))
    print(long_df["Ticker"].unique())
