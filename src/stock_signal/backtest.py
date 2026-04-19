"""
Backtesting module.
Simulates a long-only trading strategy based on model predictions
and computes portfolio performance metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def generate_signals(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    proba_threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Generate buy signals using model predicted probabilities.

    Args:
        df: Feature DataFrame.
        model: Trained XGBoost model.
        feature_cols: List of feature column names.
        proba_threshold: Minimum probability to trigger a buy signal.

    Returns:
        DataFrame with added 'proba' and 'signal' columns.
    """
    df = df.copy()
    df["proba"]  = model.predict_proba(df[feature_cols])[:, 1]
    df["signal"] = (df["proba"] >= proba_threshold).astype(int)
    print(f"Signal rate: {df['signal'].mean():.2%} ({df['signal'].sum()} signals)")
    return df


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    position_size: float = 0.1,
    holding_period: int = 5,
    trading_fee: float = 0.001,
) -> pd.DataFrame:
    """
    Simulate a long-only strategy:
    - Buy when signal == 1
    - Hold for `holding_period` days
    - Each position uses `position_size` fraction of capital
    - Apply `trading_fee` on entry and exit

    Args:
        df: DataFrame with 'signal', 'Close', 'Date', 'Ticker' columns.
        initial_capital: Starting portfolio value in USD.
        position_size: Fraction of capital per trade (default 10%).
        holding_period: Days to hold each position.
        trading_fee: Fee per trade as a fraction (default 0.1%).

    Returns:
        DataFrame of daily portfolio values.
    """
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    dates = sorted(df["Date"].unique())

    capital    = initial_capital
    positions  = {}   # {ticker: {entry_price, shares, exit_date}}
    daily_records = []

    for date in dates:
        day_df = df[df["Date"] == date]

        # --- Close positions that have reached holding period ---
        to_close = [t for t, p in positions.items() if p["exit_date"] <= date]
        for ticker in to_close:
            pos   = positions.pop(ticker)
            row   = day_df[day_df["Ticker"] == ticker]
            if row.empty:
                continue
            exit_price = row["Close"].values[0]
            proceeds   = pos["shares"] * exit_price * (1 - trading_fee)
            capital   += proceeds

        # --- Open new positions on signal ---
        signals_today = day_df[day_df["signal"] == 1]
        for _, row in signals_today.iterrows():
            ticker = row["Ticker"]
            if ticker in positions:
                continue  # already holding this stock
            if capital <= 0:
                break

            trade_capital = initial_capital * position_size
            entry_price   = row["Close"]
            shares        = (trade_capital * (1 - trading_fee)) / entry_price
            cost          = shares * entry_price * (1 + trading_fee)

            if cost > capital:
                continue

            capital -= cost
            exit_date = dates[min(dates.index(date) + holding_period, len(dates) - 1)]
            positions[ticker] = {
                "entry_price": entry_price,
                "shares":      shares,
                "exit_date":   exit_date,
            }

        # --- Mark-to-market open positions ---
        open_value = 0.0
        for ticker, pos in positions.items():
            row = day_df[day_df["Ticker"] == ticker]
            if not row.empty:
                open_value += pos["shares"] * row["Close"].values[0]

        portfolio_value = capital + open_value
        daily_records.append({"Date": date, "portfolio_value": portfolio_value})

    portfolio_df = pd.DataFrame(daily_records)
    print(f"Backtest complete | Final value: ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}")
    return portfolio_df


def compute_metrics(
    portfolio_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Compute key portfolio performance metrics.

    Args:
        portfolio_df: Daily portfolio values.
        initial_capital: Starting capital.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dictionary of metrics.
    """
    pv = portfolio_df["portfolio_value"]

    total_return   = (pv.iloc[-1] - initial_capital) / initial_capital
    daily_returns  = pv.pct_change().dropna()
    annual_return  = (1 + total_return) ** (252 / len(pv)) - 1
    annual_vol     = daily_returns.std() * np.sqrt(252)
    sharpe         = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

    rolling_max    = pv.cummax()
    drawdown       = (pv - rolling_max) / rolling_max
    max_drawdown   = drawdown.min()

    metrics = {
        "total_return":  round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        "annual_vol":    round(annual_vol * 100, 2),
        "sharpe_ratio":  round(sharpe, 4),
        "max_drawdown":  round(max_drawdown * 100, 2),
        "final_value":   round(pv.iloc[-1], 2),
    }

    print(f"\n{'='*40}")
    print("Portfolio Performance Metrics")
    print(f"{'='*40}")
    for k, v in metrics.items():
        unit = "%" if "return" in k or "vol" in k or "drawdown" in k else ""
        print(f"  {k:<18}: {v}{unit}")

    return metrics


def plot_portfolio(
    portfolio_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    save: bool = True,
) -> None:
    """
    Plot portfolio value over time vs buy-and-hold SPY benchmark.

    Args:
        portfolio_df: Daily portfolio values.
        initial_capital: Starting capital for reference line.
        save: Whether to save the chart to results/.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Portfolio value ---
    axes[0].plot(
        portfolio_df["Date"],
        portfolio_df["portfolio_value"],
        color="#2196F3", linewidth=1.8, label="Strategy"
    )
    axes[0].axhline(initial_capital, color="gray", linestyle="--", linewidth=1, label="Initial capital")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title("Stock Signal Pipeline — Backtest Results")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # --- Drawdown ---
    pv           = portfolio_df["portfolio_value"]
    rolling_max  = pv.cummax()
    drawdown     = (pv - rolling_max) / rolling_max * 100
    axes[1].fill_between(portfolio_df["Date"], drawdown, 0, color="#F44336", alpha=0.4)
    axes[1].plot(portfolio_df["Date"], drawdown, color="#F44336", linewidth=1)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "backtest_chart.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {path}")

    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from stock_signal.model import load_model, FEATURE_COLS
    from stock_signal.features import add_ticker_encoding

    df = pd.read_parquet("data/processed/features.parquet")
    df = add_ticker_encoding(df)

    model = load_model()
    df    = generate_signals(df, model, FEATURE_COLS, proba_threshold=0.55)

    portfolio_df = run_backtest(df, initial_capital=100_000)
    metrics      = compute_metrics(portfolio_df)
    plot_portfolio(portfolio_df)

    # Save outputs
    portfolio_df.to_csv(RESULTS_DIR / "portfolio_values.csv", index=False)
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "backtest_metrics.csv", index=False)
    print("Results saved.")
