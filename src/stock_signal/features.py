"""
Feature engineering module.
Computes technical indicators and labels for each ticker.
"""

import pandas as pd
import pandas_ta as ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the long-format DataFrame.
    Indicators are computed per ticker to avoid lookahead across stocks.

    Args:
        df: Long-format DataFrame with columns [Date, Ticker, Open, High, Low, Close, Volume]

    Returns:
        DataFrame with additional indicator columns.
    """
    results = []

    for ticker, group in df.groupby("Ticker"):
        group = group.copy().sort_values("Date").reset_index(drop=True)

        # --- Trend indicators ---
        group["sma_20"]  = ta.sma(group["Close"], length=20)
        group["sma_50"]  = ta.sma(group["Close"], length=50)
        group["ema_12"]  = ta.ema(group["Close"], length=12)
        group["ema_26"]  = ta.ema(group["Close"], length=26)

        # --- Momentum indicators ---
        group["rsi_14"]  = ta.rsi(group["Close"], length=14)

        macd = ta.macd(group["Close"], fast=12, slow=26, signal=9)
        group["macd"]         = macd["MACD_12_26_9"]
        group["macd_signal"]  = macd["MACDs_12_26_9"]
        group["macd_hist"]    = macd["MACDh_12_26_9"]

        # --- Volatility indicators ---
        bb = ta.bbands(group["Close"], length=20, std=2)
        group["bb_upper"] = bb["BBU_20_2.0_2.0"]
        group["bb_mid"]   = bb["BBM_20_2.0_2.0"]
        group["bb_lower"] = bb["BBL_20_2.0_2.0"]
        group["bb_width"] = (group["bb_upper"] - group["bb_lower"]) / group["bb_mid"]

        # --- Volume indicator ---
        group["obv"] = ta.obv(group["Close"], group["Volume"])

        # --- Price-derived features ---
        group["daily_return"]   = group["Close"].pct_change()
        group["return_5d"]      = group["Close"].pct_change(5)
        group["return_20d"]     = group["Close"].pct_change(20)
        group["volatility_20d"] = group["daily_return"].rolling(20).std()

        # --- Price relative to moving averages ---
        group["close_to_sma20"] = group["Close"] / group["sma_20"] - 1
        group["close_to_sma50"] = group["Close"] / group["sma_50"] - 1

        results.append(group)

    return pd.concat(results, ignore_index=True)


def add_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """
    Add a binary target label per ticker:
    1 if Close price rises more than `threshold` over `horizon` days, else 0.

    Args:
        df: Feature-enriched DataFrame.
        horizon: Number of trading days to look ahead.
        threshold: Minimum return to count as a positive signal (default 2%).

    Returns:
        DataFrame with added 'target' column.
    """
    results = []

    for ticker, group in df.groupby("Ticker"):
        group = group.copy().sort_values("Date").reset_index(drop=True)
        future_return = group["Close"].shift(-horizon) / group["Close"] - 1
        group["future_return"] = future_return
        group["target"] = (future_return > threshold).astype(int)
        results.append(group)

    return pd.concat(results, ignore_index=True)


def build_features(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """
    Full feature engineering pipeline: indicators + labels + clean NaNs.

    Args:
        df: Long-format raw OHLCV DataFrame.
        horizon: Label lookahead in trading days.
        threshold: Label threshold for positive class.

    Returns:
        Clean, feature-enriched DataFrame ready for modeling.
    """
    df = add_technical_indicators(df)
    df = add_labels(df, horizon=horizon, threshold=threshold)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN | Remaining: {after}")

    return df


if __name__ == "__main__":
    from stock_signal.ingest import load_raw, to_long_format

    raw = load_raw()
    long_df = to_long_format(raw)
    features_df = build_features(long_df)

    print(features_df.shape)
    print(features_df.columns.tolist())
    print(features_df[["Date", "Ticker", "Close", "rsi_14", "macd", "target"]].head(10))
