import streamlit as st
import pandas as pd

st.title("📊 Strategy Backtesting App")

ticker = st.text_input("Stock Ticker", "AAPL")
start  = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end    = st.date_input("End Date", pd.to_datetime("2024-01-01"))
capital = st.number_input("Initial Capital", value=10000.0)
"""
╔══════════════════════════════════════════════════════════════════════╗
║          MOVING AVERAGE CROSSOVER — BACKTESTING SYSTEM              ║
║                                                                      ║
║  Features:                                                           ║
║    • Fetches historical data via yfinance                            ║
║    • SMA crossover strategy (Golden Cross / Death Cross)             ║
║    • Full trade simulation with portfolio tracking                   ║
║    • Performance metrics: Return, Win Rate, Drawdown, Sharpe         ║
║    • 3-panel chart: Price + Signals + Equity Curve                   ║
║                                                                      ║
║  Usage:                                                              ║
║    pip install yfinance pandas numpy matplotlib                      ║
║    python backtest_strategy.py                                       ║
║                                                                      ║
║  Customise the CONFIG section below to change ticker / dates.        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf


# ══════════════════════════════════════════════════════════════════════
# CONFIG — edit these values to customise your backtest
# ══════════════════════════════════════════════════════════════════════

TICKER          = "AAPL"        # Stock symbol
START_DATE      = "2020-01-01"  # Backtest start date
END_DATE        = "2024-01-01"  # Backtest end date
INITIAL_CAPITAL = 10_000.0      # Starting capital in USD
SHORT_WINDOW    = 50            # Fast moving average period (days)
LONG_WINDOW     = 200           # Slow moving average period (days)
SAVE_CHART      = True          # Save chart as PNG?
CHART_FILENAME  = "backtest_results.png"


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Fetch Stock Data
# ══════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.

    Args:
        ticker : Stock symbol e.g. 'AAPL'
        start  : Start date as 'YYYY-MM-DD'
        end    : End date as 'YYYY-MM-DD'

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"\n[1/5] Fetching data for {ticker} ({start} → {end})...")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Fix for yfinance 0.2.x+ which returns MultiIndex columns like ('Close', 'AAPL')
    # Flatten to single-level column names: 'Close', 'Open', etc.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    print(f"      ✓ {len(df)} trading days loaded.")
    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Calculate Indicators
# ══════════════════════════════════════════════════════════════════════

def add_moving_averages(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int  = 200
) -> pd.DataFrame:
    """
    Add Simple Moving Averages to the DataFrame.

    Args:
        df           : DataFrame with a 'Close' column
        short_window : Lookback for the fast MA
        long_window  : Lookback for the slow MA

    Returns:
        DataFrame with new columns: SMA_short, SMA_long
    """
    print(f"\n[2/5] Calculating SMA({short_window}) and SMA({long_window})...")

    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(window=short_window).mean()
    df["SMA_long"]  = df["Close"].rolling(window=long_window).mean()

    valid = df["SMA_long"].notna().sum()
    print(f"      ✓ {valid} rows have both MAs calculated.")
    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Generate Trading Signals
# ══════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Buy / Sell signals using MA Crossover logic.

    Signal rules:
        +1  BUY  — short MA crosses ABOVE long MA  (Golden Cross)
        -1  SELL — short MA crosses BELOW long MA  (Death Cross)
         0  HOLD — no crossover today

    Returns:
        DataFrame with new columns: Position, Signal
    """
    print("\n[3/5] Generating crossover signals...")

    df = df.copy()
    df.dropna(subset=["SMA_short", "SMA_long"], inplace=True)

    # 1 when short MA is above long MA, else 0
    df["Position"] = (df["SMA_short"] > df["SMA_long"]).astype(int)

    # Diff captures the exact crossover day: +1 = buy, -1 = sell, 0 = hold
    df["Signal"] = df["Position"].diff().fillna(0).astype(int)

    n_buys  = (df["Signal"] ==  1).sum()
    n_sells = (df["Signal"] == -1).sum()
    print(f"      ✓ {n_buys} BUY signal(s), {n_sells} SELL signal(s) found.")
    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Backtest (Simulate Trades)
# ══════════════════════════════════════════════════════════════════════

def backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0
) -> tuple[pd.DataFrame, list]:
    """
    Simulate trades on buy/sell signals and track daily portfolio value.

    Strategy:
        BUY  — invest all available cash at closing price
        SELL — liquidate all shares at closing price
        HOLD — do nothing; portfolio value floats with price

    Args:
        df              : DataFrame with 'Close' and 'Signal' columns
        initial_capital : Starting cash in USD

    Returns:
        (DataFrame with Portfolio_Value column, list of trade dicts)
    """
    print("\n[4/5] Running backtest simulation...")

    df = df.copy()
    cash      = initial_capital
    shares    = 0.0
    trade_log = []
    portfolio_values = []

    for date, row in df.iterrows():
        price  = float(row["Close"])
        signal = int(row["Signal"])

        if signal == 1 and cash > 0:
            # BUY: go all-in
            shares = cash / price
            cash   = 0.0
            trade_log.append({
                "Date": date, "Type": "BUY",
                "Price": round(price, 2),
                "Shares": round(shares, 4)
            })

        elif signal == -1 and shares > 0:
            # SELL: liquidate everything
            cash   = shares * price
            shares = 0.0
            trade_log.append({
                "Date": date, "Type": "SELL",
                "Price": round(price, 2),
                "Value": round(cash, 2)
            })

        # Track total value (cash + market value of open position)
        portfolio_values.append(cash + shares * price)

    df["Portfolio_Value"] = portfolio_values

    final_value = cash + shares * float(df["Close"].iloc[-1])
    ret = (final_value / initial_capital - 1) * 100
    print(f"      ✓ Final portfolio value: ${final_value:,.2f}  ({ret:+.2f}%)")
    return df, trade_log


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Performance Metrics
# ══════════════════════════════════════════════════════════════════════

def calculate_metrics(
    df: pd.DataFrame,
    trade_log: list,
    initial_capital: float = 10_000.0
) -> dict:
    """
    Calculate and print key performance metrics.

    Metrics:
        Total Return     — overall % gain/loss vs starting capital
        Buy & Hold       — benchmark: simply held the stock the whole time
        Win Rate         — % of closed trades that were profitable
        Max Drawdown     — largest peak-to-trough decline in equity
        Sharpe Ratio     — annualised return-per-unit-of-risk (rf ≈ 0)

    Returns:
        dict of metric name → value
    """
    print("\n[5/5] Calculating performance metrics...")

    equity      = df["Portfolio_Value"]
    final_value = equity.iloc[-1]

    # ── Total return ──────────────────────────────────────────
    total_return = (final_value / initial_capital - 1) * 100

    # ── Buy & Hold benchmark ──────────────────────────────────
    bh_return = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100

    # ── Win rate (pair BUYs with following SELLs) ─────────────
    trades     = pd.DataFrame(trade_log)
    buy_prices = []
    wins = closed = 0

    if not trades.empty:
        for _, row in trades.iterrows():
            if row["Type"] == "BUY":
                buy_prices.append(row["Price"])
            elif row["Type"] == "SELL" and buy_prices:
                closed += 1
                if row["Price"] > buy_prices.pop(0):
                    wins += 1

    win_rate = (wins / closed * 100) if closed > 0 else 0.0

    # ── Max drawdown ──────────────────────────────────────────
    rolling_peak = equity.cummax()
    max_drawdown = ((equity - rolling_peak) / rolling_peak * 100).min()

    # ── Sharpe ratio (annualised, daily returns) ──────────────
    daily_ret    = equity.pct_change().dropna()
    sharpe_ratio = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0.0

    metrics = {
        "Initial Capital"  : initial_capital,
        "Final Value"      : final_value,
        "Total Return %"   : total_return,
        "Buy & Hold %"     : bh_return,
        "Closed Trades"    : closed,
        "Win Rate %"       : win_rate,
        "Max Drawdown %"   : max_drawdown,
        "Sharpe Ratio"     : sharpe_ratio,
    }

    # ── Pretty print ──────────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │       STRATEGY PERFORMANCE REPORT       │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Initial Capital    ${initial_capital:>15,.2f}   │")
    print(f"  │  Final Value        ${final_value:>15,.2f}   │")
    print(f"  │  Total Return       {total_return:>14.2f}%   │")
    print(f"  │  Buy & Hold         {bh_return:>14.2f}%   │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Closed Trades      {closed:>15}   │")
    print(f"  │  Win Rate           {win_rate:>14.1f}%   │")
    print(f"  │  Max Drawdown       {max_drawdown:>14.2f}%   │")
    print(f"  │  Sharpe Ratio       {sharpe_ratio:>15.2f}   │")
    print("  └─────────────────────────────────────────┘")

    return metrics


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Visualisation
# ══════════════════════════════════════════════════════════════════════

def plot_results(
    df: pd.DataFrame,
    trade_log: list,
    metrics: dict,
    ticker: str,
    save: bool = True,
    filename: str = "backtest_results.png"
) -> None:
    """
    3-panel chart:
        Panel 1 (top)    — Price + SMA lines + Buy/Sell markers + bullish shading
        Panel 2 (middle) — Volume bars
        Panel 3 (bottom) — Strategy equity curve vs Buy & Hold

    Args:
        df         : Processed DataFrame with all columns
        trade_log  : List of trade dicts from backtest()
        metrics    : Dict from calculate_metrics()
        ticker     : Stock symbol (for chart title)
        save       : Whether to save the chart to disk
        filename   : Output filename if save=True
    """
    trades = pd.DataFrame(trade_log)
    buys   = trades[trades["Type"] == "BUY"]  if not trades.empty else pd.DataFrame()
    sells  = trades[trades["Type"] == "SELL"] if not trades.empty else pd.DataFrame()

    initial_capital = metrics["Initial Capital"]
    bh_curve        = (df["Close"] / float(df["Close"].iloc[0])) * initial_capital

    # ── Figure setup ──────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 10),
        gridspec_kw={"height_ratios": [3, 1, 2]},
        sharex=True
    )
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#888888", labelsize=9)
        ax.yaxis.label.set_color("#888888")
        ax.grid(axis="y", color="#222222", linewidth=0.5, linestyle="--")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")

    # ── Panel 1: Price + MAs + Signals ───────────────────────
    ax1 = axes[0]

    ax1.plot(df.index, df["Close"],
             color="#d4d4d4", lw=1, label="Close price", alpha=0.9)
    ax1.plot(df.index, df["SMA_short"],
             color="#f5c542", lw=1.5, label=f"SMA {SHORT_WINDOW}", alpha=0.95)
    ax1.plot(df.index, df["SMA_long"],
             color="#5b9cf6", lw=1.5, label=f"SMA {LONG_WINDOW}", alpha=0.95)

    # Bullish zone shading (short MA above long MA)
    ax1.fill_between(
        df.index, df["SMA_short"], df["SMA_long"],
        where=(df["SMA_short"] > df["SMA_long"]),
        alpha=0.07, color="#00e676"
    )

    # Buy markers
    if not buys.empty:
        ax1.scatter(
            buys["Date"], buys["Price"],
            marker="^", color="#00e676", s=130,
            zorder=5, label="BUY signal"
        )

    # Sell markers
    if not sells.empty:
        ax1.scatter(
            sells["Date"], sells["Price"],
            marker="v", color="#ff4d4d", s=130,
            zorder=5, label="SELL signal"
        )

    ax1.set_ylabel("Price (USD)", color="#888888")
    ax1.legend(
        loc="upper left", fontsize=8.5,
        facecolor="#1a1a1a", labelcolor="#cccccc",
        framealpha=0.9, edgecolor="#333333"
    )
    ax1.set_title(
        f"{ticker}  ·  MA Crossover Strategy  ·  "
        f"Return: {metrics['Total Return %']:+.1f}%   "
        f"B&H: {metrics['Buy & Hold %']:+.1f}%   "
        f"Max DD: {metrics['Max Drawdown %']:.1f}%   "
        f"Sharpe: {metrics['Sharpe Ratio']:.2f}",
        color="#eeeeee", fontsize=10, pad=12,
        fontweight="bold"
    )

    # ── Panel 2: Volume ───────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(df.index, df["Volume"], color="#2e2e2e", width=1.2, alpha=0.9)
    ax2.set_ylabel("Volume", color="#888888")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M")
    )

    # ── Panel 3: Equity Curve vs Buy & Hold ──────────────────
    ax3 = axes[2]

    ax3.plot(df.index, df["Portfolio_Value"],
             color="#00e676", lw=1.8, label="Strategy equity")
    ax3.plot(df.index, bh_curve,
             color="#5b9cf6", lw=1.5, label="Buy & Hold", linestyle="--", alpha=0.8)
    ax3.axhline(
        initial_capital, color="#555555",
        lw=0.8, linestyle=":", label="Starting capital"
    )

    # Profit / loss shading
    ax3.fill_between(
        df.index, df["Portfolio_Value"], initial_capital,
        where=(df["Portfolio_Value"] >= initial_capital),
        alpha=0.08, color="#00e676"
    )
    ax3.fill_between(
        df.index, df["Portfolio_Value"], initial_capital,
        where=(df["Portfolio_Value"] < initial_capital),
        alpha=0.12, color="#ff4d4d"
    )

    ax3.set_ylabel("Portfolio Value (USD)", color="#888888")
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax3.legend(
        loc="upper left", fontsize=8.5,
        facecolor="#1a1a1a", labelcolor="#cccccc",
        framealpha=0.9, edgecolor="#333333"
    )

    # ── Shared x-axis ─────────────────────────────────────────
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=30, ha="right", color="#888888", fontsize=9)

    plt.tight_layout(h_pad=0.4)

    if save:
        plt.savefig(filename, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n  Chart saved → {filename}")

    plt.show()


# ══════════════════════════════════════════════════════════════════════
# MAIN — Run the full pipeline
# ══════════════════════════════════════════════════════════════════════
# ================================
# UI + RUN BACKTEST
# ================================

st.title("📊 Strategy Backtesting App")

ticker = st.text_input("Stock Ticker", "AAPL")
start = st.date_input("Start Date")
end = st.date_input("End Date")
capital = st.number_input("Initial Capital", value=10000.0)

# 🔥 Run Button
if st.button("🚀 Run Backtest"):
    data = fetch_stock_data(ticker, str(start), str(end))

    if data.empty:
        st.error("No data found. Try another stock or date range.")
        st.stop()

    data = add_moving_averages(data)
    data = generate_signals(data)
    data, trades = backtest(data, capital)
    metrics = calculate_metrics(data, trades, capital)

    st.write("### 📈 Performance")
    st.json(metrics)

    plot_results(data, trades, metrics, ticker)




  
