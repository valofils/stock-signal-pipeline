"""
Daily pipeline runner.
Orchestrates: fetch -> features -> predict -> report.
Run this script daily after market close.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from stock_signal.ingest import fetch_historical, save_raw, to_long_format, TICKERS
from stock_signal.features import build_features, add_ticker_encoding
from stock_signal.model import load_model, FEATURE_COLS
from stock_signal.backtest import generate_signals
from stock_signal.report import generate_report

DATA_DIR    = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def run():
    print("\n" + "="*55)
    print(f"  Stock Signal Pipeline -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55 + "\n")

    print("[ 1/4 ] Fetching latest market data...")
    raw = fetch_historical(tickers=TICKERS, period="1y", interval="1d")
    save_raw(raw, filename="ohlcv_latest.parquet")
    long_df = to_long_format(raw)

    print("\n[ 2/4 ] Computing features and indicators...")
    features_df = build_features(long_df)
    features_df = add_ticker_encoding(features_df)
    out_path = DATA_DIR / "processed" / "features_latest.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out_path)
    print(f"Features saved | shape: {features_df.shape}")

    print("\n[ 3/4 ] Generating signals with trained model...")
    model = load_model("xgb_model_oos.joblib")
    features_df = generate_signals(features_df, model, FEATURE_COLS, proba_threshold=0.55)

    latest_date = features_df["Date"].max()
    todays_signals = (
        features_df[
            (features_df["Date"] == latest_date) &
            (features_df["signal"] == 1)
        ][["Date", "Ticker", "Close", "proba", "rsi_14", "macd", "return_5d"]]
        .sort_values("proba", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nSignals for {latest_date.date()} ({len(todays_signals)} stocks):")
    print(todays_signals.to_string(index=False))

    print("\n[ 4/4 ] Generating HTML report...")
    report_path = generate_report(todays_signals, features_df, latest_date)

    print("\n" + "="*55)
    print(f"  Pipeline complete. Report: {report_path}")
    print("="*55 + "\n")


if __name__ == "__main__":
    run()
