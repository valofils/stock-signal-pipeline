# Stock Signal Pipeline

An end-to-end automated stock signal pipeline that fetches daily market data, computes technical indicators, generates buy signals using a machine learning model, and publishes a daily HTML report via GitHub Actions.

---

## Architecture

```
yfinance API --> Feature Engineering --> XGBoost Classifier --> Signal Report
     |                                        |                       |
Daily OHLCV         RSI, MACD,          Binary classifier    HTML report committed
15 S&P 500        Bollinger Bands,      predicts 5-day       to repo every weekday
 tickers          SMAs, OBV, etc.       price movement       at 4pm EST via cron
```

## Project Structure

```
stock-signal-pipeline/
├── .github/workflows/daily_pipeline.yml  # GitHub Actions cron job
├── src/stock_signal/
│   ├── ingest.py          # Data fetching via yfinance
│   ├── features.py        # Technical indicator computation
│   ├── model.py           # XGBoost training with walk-forward validation
│   ├── backtest.py        # Trade simulation and portfolio metrics
│   ├── report.py          # HTML report generation
│   └── run_pipeline.py    # Daily pipeline orchestrator
├── notebooks/             # Exploratory analysis notebooks
├── data/
│   ├── raw/               # Raw OHLCV parquet files
│   └── processed/         # Feature-engineered parquet files
└── results/
    ├── reports/           # Daily HTML signal reports
    ├── backtest_chart.png
    └── backtest_metrics_oos.csv
```

---

## Features

**Data**
- 15 S&P 500 stocks across 6 sectors (Technology, Finance, Healthcare, Energy, Consumer, Industrials)
- 2 years of daily OHLCV data via yfinance (free, no API key required)
- Stored as Parquet for efficient read/write

**Technical Indicators**
- Trend: SMA(20), SMA(50), EMA(12), EMA(26)
- Momentum: RSI(14), MACD(12,26,9)
- Volatility: Bollinger Bands(20,2), Bollinger Band Width
- Volume: On-Balance Volume (OBV)
- Price-derived: 1d/5d/20d returns, 20d rolling volatility, price-to-SMA ratios

**Model**
- Algorithm: XGBoost binary classifier
- Target: Will this stock rise more than 2% over the next 5 trading days?
- Validation: Walk-forward cross-validation (5 folds) to prevent data leakage
- Class imbalance handled via `scale_pos_weight`

**Backtesting**
- Long-only strategy: buy on signal, hold for 5 days
- Position sizing: 10% of capital per trade
- Trading fees: 0.1% per transaction
- Metrics: total return, annualised return, Sharpe ratio, max drawdown

---

## Results

Out-of-sample backtest (Dec 2025 – Apr 2026, truly unseen data):

| Metric               | Strategy | SPY Benchmark |
|----------------------|----------|---------------|
| Total Return         | -1.91%   | +0.52%        |
| Annualised Return    | -5.26%   | —             |
| Annualised Volatility| 10.83%   | —             |
| Sharpe Ratio         | -0.95    | —             |
| Max Drawdown         | -9.47%   | —             |

Walk-forward validation mean ROC-AUC: **0.55** across 4 folds.

> **Note on data leakage:** An initial backtest on the full dataset produced an inflated 151% return
> because the model had seen the test period during training. This was identified, corrected, and
> documented. The OOS results above are the honest numbers. Catching and fixing this is part of the story.

---

## Automation

The pipeline runs automatically every weekday at 21:00 UTC (4pm EST) via GitHub Actions:

1. Fetch latest daily OHLCV data for all 15 tickers
2. Compute technical indicators and features
3. Generate buy signals using the trained XGBoost model
4. Produce an HTML report saved to `results/reports/`
5. Commit and push the report back to the repository

View the latest report: [`results/reports/`](results/reports/)

---

## Running Locally

**Requirements:** Python 3.10+

```bash
# Clone and set up
git clone https://github.com/valofils/stock-signal-pipeline.git
cd stock-signal-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run the full pipeline
python src/stock_signal/run_pipeline.py

# Or run individual modules
python src/stock_signal/ingest.py      # Fetch data
python src/stock_signal/features.py   # Compute indicators
python src/stock_signal/model.py      # Train model
python src/stock_signal/backtest.py   # Run backtest
```

---

## Key Technical Decisions

**Walk-forward validation over random split**
Standard train/test splits on time-series data leak future information into training. Walk-forward validation ensures the model is always evaluated on data it has never seen, mimicking real-world deployment conditions.

**Out-of-sample backtesting**
The final backtest uses only the last 20% of data, held out entirely from model training, to simulate realistic forward performance. The inflated in-sample results are documented as a learning point about data leakage.

**Parquet over CSV**
Raw and processed data are stored as Parquet files for efficient columnar storage, faster reads, and native support for datetime dtypes — important for time-series pipelines.

**GitHub Actions for automation**
The pipeline is scheduled via cron in GitHub Actions rather than a local cron job, making it portable, auditable, and runnable without a persistent server.

---

## Tech Stack

| Tool          | Purpose                                |
|---------------|----------------------------------------|
| yfinance      | Market data ingestion                  |
| pandas-ta     | Technical indicator computation        |
| XGBoost       | Signal classification model            |
| joblib        | Model serialisation                    |
| matplotlib    | Backtest visualisation                 |
| GitHub Actions| Pipeline scheduling and automation     |
| pyarrow       | Efficient Parquet storage              |

---

## Limitations and Next Steps

- Model ROC-AUC of ~0.55 indicates a weak but non-random signal; per-ticker models or richer features could improve this
- Strategy is long-only; a market-neutral version (long strong signals, short weak ones) would reduce market exposure
- Threshold tuning and position sizing optimisation not yet explored
- Alpaca paper trading API integration planned to simulate live trade execution

---

## Disclaimer

This project is for educational and portfolio purposes only. It is not financial advice and should not be used to make real investment decisions.

---

Built by [Mariel](https://github.com/valofils) · Inspired by the [DataTalksClub Stock Markets Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp)
