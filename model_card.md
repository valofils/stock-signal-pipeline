# Model Card: Stock Signal Classifier

## Model Overview

| Field | Details |
|---|---|
| **Model name** | Stock Signal XGBoost Classifier |
| **Version** | 0.1.0 |
| **Type** | Binary classification |
| **Algorithm** | XGBoost (Gradient Boosted Decision Trees) |
| **Task** | Predict whether a stock will rise more than 2% over the next 5 trading days |
| **Output** | Probability score [0, 1]; signal = 1 if probability ≥ 0.55 |
| **Developed by** | Mariel (github.com/valofils) |
| **Last updated** | April 2026 |
| **License** | MIT |

---

## Intended Use

**Primary use case:** Generate daily buy signals for a basket of 15 S&P 500 stocks as part of an automated trading pipeline.

**Intended users:** Data science portfolio demonstration. Not intended for real investment decisions.

**Out-of-scope uses:**
- Live trading with real capital
- Stocks outside the 15-ticker training basket
- Intraday or high-frequency signal generation
- Any financial decision-making

---

## Training Data

| Field | Details |
|---|---|
| **Source** | Yahoo Finance via `yfinance` library |
| **Universe** | 15 S&P 500 stocks across 6 sectors |
| **Period** | ~2 years of daily OHLCV data |
| **Training window** | First 80% of available dates (approx. Jun 2024 – Nov 2025) |
| **Frequency** | Daily (1d interval) |
| **Rows** | ~5,355 after feature engineering and NaN removal |

**Ticker universe:**

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL |
| Finance | JPM, BAC, GS |
| Healthcare | JNJ, PFE, UNH |
| Energy | XOM, CVX |
| Consumer | AMZN, WMT |
| Industrials | CAT, BA |

**Label definition:**
```
target = 1  if  Close[t+5] / Close[t] - 1  >  0.02
target = 0  otherwise
```
A 2% threshold over 5 trading days was chosen to filter out noise and focus on meaningful price moves.

**Class distribution (training set):**
- Positive (buy signal): ~33%
- Negative (no signal): ~67%
- `scale_pos_weight = 2.04` applied to compensate for imbalance

---

## Features

| Feature | Description |
|---|---|
| `ticker_encoded` | Numeric encoding of ticker symbol (0–14) |
| `sma_20` | 20-day simple moving average |
| `sma_50` | 50-day simple moving average |
| `ema_12` | 12-day exponential moving average |
| `ema_26` | 26-day exponential moving average |
| `rsi_14` | 14-day Relative Strength Index |
| `macd` | MACD line (EMA12 – EMA26) |
| `macd_signal` | 9-day EMA of MACD |
| `macd_hist` | MACD histogram (MACD – signal) |
| `bb_upper` | Bollinger Band upper (20d, 2 std) |
| `bb_mid` | Bollinger Band middle (20d SMA) |
| `bb_lower` | Bollinger Band lower (20d, 2 std) |
| `bb_width` | Bollinger Band width (upper – lower) / mid |
| `obv` | On-Balance Volume |
| `daily_return` | 1-day price return |
| `return_5d` | 5-day price return |
| `return_20d` | 20-day price return |
| `volatility_20d` | 20-day rolling standard deviation of daily returns |
| `close_to_sma20` | Close / SMA20 – 1 |
| `close_to_sma50` | Close / SMA50 – 1 |

All features are computed per ticker independently to avoid cross-stock contamination.

---

## Model Architecture

```
XGBClassifier(
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = 2.04,
    eval_metric      = "logloss",
    random_state     = 42,
)
```

The final production model is trained on the full training set (first 80% of dates) after hyperparameters are validated via walk-forward cross-validation.

---

## Evaluation

### Validation Method: Walk-Forward Cross-Validation

Standard k-fold cross-validation is inappropriate for time-series data because it allows future information to leak into training. Walk-forward validation is used instead:

```
Fold 1: Train [t0 → t1]         Test [t1 → t2]
Fold 2: Train [t0 → t2]         Test [t2 → t3]
Fold 3: Train [t0 → t3]         Test [t3 → t4]
Fold 4: Train [t0 → t4]         Test [t4 → t5]
```

Each fold trains on all available history up to a cutoff point and tests on the subsequent unseen window. This strictly prevents data leakage.

### Walk-Forward Results (4 folds on training set)

| Fold | Train Size | Test Size | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 1 | 1,065 | 1,065 | 0.508 | 0.280 | 0.322 | 0.299 |
| 2 | 2,130 | 1,065 | 0.590 | 0.357 | 0.497 | 0.416 |
| 3 | 3,195 | 1,065 | 0.558 | 0.431 | 0.264 | 0.327 |
| 4 | 4,260 | 1,065 | 0.534 | 0.343 | 0.333 | 0.346 |
| **Mean** | | | **0.548** | **0.353** | **0.354** | **0.347** |

### Out-of-Sample Backtest (Dec 2025 – Apr 2026)

The final model was evaluated on the last 20% of data — a period completely unseen during training or validation.

| Metric | Value |
|---|---|
| Total Return | -1.91% |
| SPY Benchmark Return | +0.52% |
| Annualised Return | -5.26% |
| Annualised Volatility | 10.83% |
| Sharpe Ratio | -0.95 |
| Max Drawdown | -9.47% |
| Signal Rate | ~25% of trading days |

---

## Data Leakage Incident and Correction

During development, an initial backtest on the full dataset (train + test) produced a total return of **+151%** and a Sharpe ratio of **6.34**. These results were identified as invalid because the model had been trained on the same data it was being tested on — a form of data leakage common in naive ML backtests.

**Correction applied:**
- The dataset was split 80/20 by date before any model training
- The model was retrained exclusively on the first 80% of dates
- The backtest was rerun on the held-out 20% only
- Results dropped to -1.91%, which reflects true out-of-sample performance

This correction is documented here as a deliberate part of the project to demonstrate awareness of a critical pitfall in financial ML.

---

## Limitations

- **Weak signal:** Mean ROC-AUC of 0.55 is only marginally above random. The model generates directional signal but not a strong one.
- **Single model for all tickers:** One global model is trained across all 15 stocks. Per-ticker models may capture stock-specific dynamics better.
- **No fundamental data:** The model uses only price and volume-derived features. Earnings, macro indicators, and sentiment are not included.
- **Fixed threshold:** The 0.55 probability threshold and 2% label threshold are not optimised; tuning these may improve precision.
- **Look-ahead in label construction:** The `future_return` label uses prices 5 days ahead. Care is taken to ensure this column is never used as an input feature, only as the target.
- **Market regime sensitivity:** The model is trained on a specific 18-month window. Performance may degrade significantly in different market regimes.
- **Long-only strategy:** The backtesting strategy only takes long positions. In a bear market, this means no mechanism to profit from or hedge against falling prices.

---

## Ethical Considerations

- This model is trained and evaluated on publicly available market data. No private or proprietary data is used.
- The model should not be used for real financial decisions. Stock market prediction is inherently uncertain and past performance does not guarantee future results.
- Signal outputs should be treated as one of many inputs in a broader investment process, not as actionable advice.

---

## How to Retrain

```bash
# From the project root with venv activated
python src/stock_signal/ingest.py       # Refresh raw data
python src/stock_signal/features.py     # Recompute features
python src/stock_signal/model.py        # Retrain and save model
python src/stock_signal/backtest.py     # Re-evaluate
```

The trained model is saved to `results/xgb_model_oos.joblib` and loaded automatically by the daily pipeline.

---

## Citation

```
Stock Signal Pipeline
Author: Mariel (github.com/valofils)
Repository: https://github.com/valofils/stock-signal-pipeline
Year: 2026
```