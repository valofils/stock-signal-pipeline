import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from stock_signal.ingest import load_raw, to_long_format
from stock_signal.features import build_features, add_ticker_encoding
from stock_signal.model import load_model, FEATURE_COLS
from stock_signal.backtest import generate_signals, compute_metrics

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Signal Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
  .metric-positive { color: #2e7d32; }
  .metric-negative { color: #c62828; }
  .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading market data...")
def load_data():
    raw        = load_raw()
    long_df    = to_long_format(raw)
    features   = build_features(long_df)
    features   = add_ticker_encoding(features)
    return features

@st.cache_resource(show_spinner="Loading model...")
def load_trained_model():
    return load_model("xgb_model_oos.joblib")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Signal Pipeline")
    st.caption("Automated ML-powered trading signals")
    st.divider()

    df      = load_data()
    model   = load_trained_model()
    tickers = sorted(df["Ticker"].unique())

    st.subheader("Settings")
    selected_ticker = st.selectbox("Ticker", tickers, index=tickers.index("AAPL"))
    threshold = st.slider("Signal threshold", 0.50, 0.80, 0.55, 0.01,
                          help="Minimum model probability to trigger a buy signal")
    date_range = st.date_input("Date range",
        value=[df["Date"].min().date(), df["Date"].max().date()],
        min_value=df["Date"].min().date(),
        max_value=df["Date"].max().date())

    st.divider()
    st.caption(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}")
    st.caption(f"Tickers: {len(tickers)} S&P 500 stocks")
    st.caption("Model: XGBoost · Walk-forward validated")
    st.caption("Not financial advice.")


# ── Generate signals ──────────────────────────────────────────────────────────
df_signals = generate_signals(df, model, FEATURE_COLS, proba_threshold=threshold)

# Apply date filter
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df_filtered = df_signals[(df_signals["Date"] >= start) & (df_signals["Date"] <= end)]
else:
    df_filtered = df_signals

latest_date   = df_signals["Date"].max()
todays        = df_signals[df_signals["Date"] == latest_date]
todays_buy    = todays[todays["signal"] == 1].sort_values("proba", ascending=False)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Signal Pipeline")
st.caption(f"Latest data: **{latest_date.date()}** · Signal threshold: **{threshold:.0%}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickers tracked", len(tickers))
col2.metric("Buy signals today", len(todays_buy))
col3.metric("Signal rate (all time)",
            f"{df_signals['signal'].mean():.1%}")
col4.metric("Latest close (selected)",
            f"${todays[todays['Ticker']==selected_ticker]['Close'].values[0]:.2f}"
            if not todays[todays["Ticker"]==selected_ticker].empty else "N/A")

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Today's Signals",
    "📊 Price & Indicators",
    "💼 Backtest Results",
    "🧠 Model Info",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TODAY'S SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Buy Signals for {latest_date.date()}")

    if todays_buy.empty:
        st.info("No buy signals today at the current threshold. Try lowering the slider.")
    else:
        display_cols = ["Ticker", "Close", "proba", "rsi_14", "macd", "return_5d", "volatility_20d"]
        display = todays_buy[display_cols].copy()
        display.columns = ["Ticker", "Price ($)", "Confidence", "RSI (14)", "MACD", "5d Return", "20d Vol"]
        display["Price ($)"]  = display["Price ($)"].map("${:.2f}".format)
        display["Confidence"] = display["Confidence"].map("{:.1%}".format)
        display["RSI (14)"]   = display["RSI (14)"].map("{:.1f}".format)
        display["MACD"]       = display["MACD"].map("{:.3f}".format)
        display["5d Return"]  = display["5d Return"].map("{:+.2%}".format)
        display["20d Vol"]    = display["20d Vol"].map("{:.3f}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Signal Rate by Ticker (all time)")
    sig_rate = (df_signals.groupby("Ticker")["signal"].mean()
                .sort_values(ascending=False).reset_index())
    fig = px.bar(sig_rate, x="Ticker", y="signal",
                 color="signal", color_continuous_scale="Blues",
                 labels={"signal": "Signal Rate"})
    fig.update_layout(height=350, showlegend=False,
                      yaxis_tickformat=".0%", plot_bgcolor="white")
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Signal History — Selected Ticker")
    df_ticker = df_filtered[df_filtered["Ticker"] == selected_ticker].sort_values("Date")
    buy_days  = df_ticker[df_ticker["signal"] == 1]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_ticker["Date"], y=df_ticker["Close"],
        mode="lines", name="Close", line=dict(color="#2196F3", width=1.8)))
    fig2.add_trace(go.Scatter(x=buy_days["Date"], y=buy_days["Close"],
        mode="markers", name="Buy Signal",
        marker=dict(color="#4CAF50", size=9, symbol="triangle-up",
                    line=dict(color="white", width=1))))
    fig2.update_layout(height=380, plot_bgcolor="white",
                       title=f"{selected_ticker} — Price with Buy Signals",
                       xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRICE & INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    df_t = df_filtered[df_filtered["Ticker"] == selected_ticker].sort_values("Date")
    st.subheader(f"{selected_ticker} — Price, Bollinger Bands & RSI")

    fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=["Price + Bollinger Bands", "RSI (14)", "MACD"])

    # Price + BB
    fig3.add_trace(go.Scatter(x=df_t["Date"], y=df_t["bb_upper"],
        line=dict(color="rgba(100,100,200,0.3)", width=1), name="BB Upper", showlegend=False), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df_t["Date"], y=df_t["bb_lower"],
        fill="tonexty", fillcolor="rgba(100,100,200,0.07)",
        line=dict(color="rgba(100,100,200,0.3)", width=1), name="BB Lower", showlegend=False), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df_t["Date"], y=df_t["Close"],
        line=dict(color="#2196F3", width=2), name="Close"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df_t["Date"], y=df_t["sma_20"],
        line=dict(color="#FF9800", width=1.2, dash="dash"), name="SMA 20"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df_t["Date"], y=df_t["sma_50"],
        line=dict(color="#9C27B0", width=1.2, dash="dot"), name="SMA 50"), row=1, col=1)

    # RSI
    fig3.add_trace(go.Scatter(x=df_t["Date"], y=df_t["rsi_14"],
        line=dict(color="#E91E63", width=1.5), name="RSI"), row=2, col=1)
    fig3.add_hline(y=70, line_dash="dash", line_color="red",   line_width=1, row=2, col=1)
    fig3.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)

    # MACD
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in df_t["macd_hist"]]
    fig3.add_trace(go.Bar(x=df_t["Date"], y=df_t["macd_hist"],
        marker_color=colors, name="MACD Hist", showlegend=False), row=3, col=1)
    fig3.add_trace(go.Scatter(x=df_t["Date"], y=df_t["macd"],
        line=dict(color="#2196F3", width=1.2), name="MACD"), row=3, col=1)
    fig3.add_trace(go.Scatter(x=df_t["Date"], y=df_t["macd_signal"],
        line=dict(color="#FF9800", width=1.2), name="Signal"), row=3, col=1)

    fig3.update_layout(height=650, plot_bgcolor="white",
                       legend=dict(orientation="h", y=1.02))
    fig3.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Data")
        recent = df_t[["Date","Close","rsi_14","macd","bb_width","return_5d"]].tail(10).copy()
        recent.columns = ["Date","Close","RSI","MACD","BB Width","5d Return"]
        recent["Close"]     = recent["Close"].map("${:.2f}".format)
        recent["RSI"]       = recent["RSI"].map("{:.1f}".format)
        recent["5d Return"] = recent["5d Return"].map("{:+.2%}".format)
        st.dataframe(recent, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Return Distributions")
        fig4 = go.Figure()
        for col_, label_, color_ in [
                ("daily_return", "Daily", "#2196F3"),
                ("return_5d",    "5-Day", "#4CAF50"),
                ("return_20d",   "20-Day","#FF9800")]:
            fig4.add_trace(go.Histogram(x=df_t[col_]*100, name=label_,
                opacity=0.6, nbinsx=40, marker_color=color_))
        fig4.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
        fig4.update_layout(barmode="overlay", height=320,
                           plot_bgcolor="white", xaxis_title="Return (%)")
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Out-of-Sample Backtest (Dec 2025 – Apr 2026)")

    portfolio_path = Path("results/portfolio_values_oos.csv")
    metrics_path   = Path("results/backtest_metrics_oos.csv")

    if portfolio_path.exists() and metrics_path.exists():
        port_df = pd.read_csv(portfolio_path, parse_dates=["Date"])
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return",      f"{metrics['total_return']:.2f}%",
                    delta=f"SPY: +0.52%")
        col2.metric("Annual Return",     f"{metrics['annual_return']:.2f}%")
        col3.metric("Annual Volatility", f"{metrics['annual_vol']:.2f}%")
        col4.metric("Sharpe Ratio",      f"{metrics['sharpe_ratio']:.2f}")
        col5.metric("Max Drawdown",      f"{metrics['max_drawdown']:.2f}%")

        # Portfolio chart
        pv = port_df["portfolio_value"]
        rolling_max = pv.cummax()
        drawdown    = (pv - rolling_max) / rolling_max * 100

        fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=["Portfolio Value", "Drawdown (%)"])

        fig5.add_trace(go.Scatter(x=port_df["Date"], y=port_df["portfolio_value"],
            line=dict(color="#2196F3", width=2), name="Strategy"), row=1, col=1)
        fig5.add_hline(y=100000, line_dash="dash", line_color="gray",
                       line_width=1, annotation_text="Initial capital", row=1, col=1)
        fig5.add_trace(go.Scatter(x=port_df["Date"], y=drawdown,
            fill="tozeroy", fillcolor="rgba(244,67,54,0.2)",
            line=dict(color="#F44336", width=1.2), name="Drawdown"), row=2, col=1)

        fig5.update_layout(height=480, plot_bgcolor="white", showlegend=False)
        fig5.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig5, use_container_width=True)

        st.info("Note: The initial backtest on full data showed +151% return due to data leakage. "
                "This OOS backtest uses only data the model never saw during training.")
    else:
        st.warning("Backtest results not found. Run `python src/stock_signal/backtest.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model: XGBoost Binary Classifier")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Task:** Predict if a stock rises >2% over the next 5 trading days

**Algorithm:** XGBoost (Gradient Boosted Trees)

**Validation:** Walk-forward cross-validation (4 folds)

**Mean ROC-AUC:** 0.548

**Features:** 20 technical indicators

**Label threshold:** 2% over 5 days

**Signal threshold:** Configurable (sidebar slider)
        """)
    with col2:
        params = {
            "n_estimators": 200, "max_depth": 4,
            "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "scale_pos_weight": 2.04,
        }
        st.markdown("**Hyperparameters:**")
        st.json(params)

    st.subheader("Feature Importances")
    importances = model.feature_importances_
    feat_df = (pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
               .sort_values("Importance", ascending=True))
    fig6 = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Blues")
    fig6.update_layout(height=500, plot_bgcolor="white", showlegend=False,
                       yaxis_title="", xaxis_title="Feature Importance (gain)")
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Walk-Forward Validation Results")
    wf_data = {
        "Fold": [1, 2, 3, 4],
        "Train Size": [1065, 2130, 3195, 4260],
        "Test Size":  [1065, 1065, 1065, 1065],
        "ROC-AUC":    [0.508, 0.590, 0.558, 0.534],
        "Precision":  [0.280, 0.357, 0.431, 0.343],
        "Recall":     [0.322, 0.497, 0.264, 0.333],
        "F1":         [0.299, 0.416, 0.327, 0.346],
    }
    wf_df = pd.DataFrame(wf_data)
    st.dataframe(wf_df, use_container_width=True, hide_index=True)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=wf_df["Fold"], y=wf_df["ROC-AUC"],
        mode="lines+markers", name="ROC-AUC",
        line=dict(color="#2196F3", width=2), marker=dict(size=8)))
    fig7.add_trace(go.Scatter(x=wf_df["Fold"], y=wf_df["F1"],
        mode="lines+markers", name="F1",
        line=dict(color="#4CAF50", width=2), marker=dict(size=8)))
    fig7.add_hline(y=0.5, line_dash="dash", line_color="gray",
                   line_width=1, annotation_text="Random baseline")
    fig7.update_layout(height=320, plot_bgcolor="white",
                       xaxis_title="Fold", yaxis_title="Score",
                       title="Metrics Across Walk-Forward Folds")
    st.plotly_chart(fig7, use_container_width=True)
