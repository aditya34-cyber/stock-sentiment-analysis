import re
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import streamlit as st

from datetime import date, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ALPHAVANTAGE_API_KEY = "U0EAQK4B9887LG82"
SENTIMENT_SMOOTH_WINDOW = 5   # rolling window for sentiment smoothing
BACKTEST_FORWARD_DAYS = 5     # forward-looking window for backtesting

analyzer = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .metric-label { font-size: 0.85rem !important; }
        .signal-buy   { background:#0d3b0d; color:#4cff6e; padding:0.4rem 1.2rem;
                        border-radius:8px; font-weight:700; font-size:1.1rem; display:inline-block; }
        .signal-sell  { background:#3b0d0d; color:#ff4c4c; padding:0.4rem 1.2rem;
                        border-radius:8px; font-weight:700; font-size:1.1rem; display:inline-block; }
        .signal-hold  { background:#2a2a0d; color:#ffe74c; padding:0.4rem 1.2rem;
                        border-radius:8px; font-weight:700; font-size:1.1rem; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Stock Sentiment Analyzer")
st.caption("Sentiment-driven stock analysis — powered by VADER NLP + Alpha Vantage + yFinance")


# ─────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip URLs, special chars, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)      # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# SENTIMENT FETCHING
# ─────────────────────────────────────────────

def fetch_news_sentiment_alphavantage(symbol: str) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    params = {"function": "NEWS_SENTIMENT", "tickers": symbol, "apikey": ALPHAVANTAGE_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception as e:
        st.warning(f"Alpha Vantage request failed: {e}")
        return pd.DataFrame(columns=["date", "sentiment"])

    articles = data.get("feed", [])
    if not articles:
        return pd.DataFrame(columns=["date", "sentiment"])

    rows = []
    for article in articles:
        date_str = article.get("time_published", "")
        try:
            date_value = pd.to_datetime(date_str).date()
        except Exception:
            continue
        title   = clean_text(article.get("title", ""))
        summary = clean_text(article.get("summary", ""))
        text    = f"{title}. {summary}"
        score   = analyzer.polarity_scores(text)["compound"]
        rows.append({"date": date_value, "sentiment": score})

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment"])

    df = pd.DataFrame(rows)
    return df.groupby("date")["sentiment"].mean().reset_index()


def fetch_news_sentiment_yfinance(symbol: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(symbol)
        news = getattr(stock, "news", [])
    except Exception:
        return pd.DataFrame(columns=["date", "sentiment"])

    rows = []
    for article in news:
        ts = article.get("providerPublishTime")
        if not ts:
            continue
        date_value = pd.to_datetime(ts, unit="s").date()
        title   = clean_text(article.get("title", ""))
        summary = clean_text(article.get("summary", ""))
        score   = analyzer.polarity_scores(f"{title}. {summary}")["compound"]
        rows.append({"date": date_value, "sentiment": score})

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment"])

    df = pd.DataFrame(rows)
    return df.groupby("date")["sentiment"].mean().reset_index()


def fetch_news_sentiment(symbol: str) -> pd.DataFrame:
    """Try Alpha Vantage first; fall back to yFinance."""
    df = fetch_news_sentiment_alphavantage(symbol)
    if df.empty:
        st.info("Alpha Vantage returned no data — falling back to yFinance news.")
        df = fetch_news_sentiment_yfinance(symbol)
    return df


# ─────────────────────────────────────────────
# PRICE FETCHING
# ─────────────────────────────────────────────

def fetch_historical_data(symbol: str, start_date: date) -> pd.DataFrame:
    raw = yf.download(symbol, start=start_date, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw["date"] = pd.to_datetime(raw["Date"]).dt.date
    return raw[["date", "Close", "Open", "High", "Low", "Volume"]].copy()


# ─────────────────────────────────────────────
# COMBINE & SMOOTH
# ─────────────────────────────────────────────

def build_combined_df(symbol: str) -> pd.DataFrame:
    """Merge price data with sentiment, apply smoothing."""
    news_df = fetch_news_sentiment(symbol)
    if news_df.empty:
        st.warning("No sentiment data found for this stock.")
        return pd.DataFrame()

    # Align sentiment date: shift by -1 day so it leads price
    news_df["date"] = (pd.to_datetime(news_df["date"]) - pd.Timedelta(days=1)).dt.date

    start = date.today() - timedelta(days=730)
    price_df = fetch_historical_data(symbol, start_date=start)
    if price_df.empty:
        st.warning("No price data retrieved.")
        return pd.DataFrame()

    df = pd.merge(price_df, news_df, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # Fill missing sentiment with forward-fill then 0
    df["sentiment"] = df["sentiment"].fillna(method="ffill", limit=3).fillna(0)

    # Smoothed sentiment (rolling mean)
    df["sentiment_smooth"] = (
        df["sentiment"].rolling(window=SENTIMENT_SMOOTH_WINDOW, min_periods=1).mean()
    )

    # Daily return for backtesting
    df["daily_return"] = df["Close"].pct_change()

    return df


# ─────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────

def compute_signal(df: pd.DataFrame) -> str:
    """
    BUY  — smoothed sentiment > 0.05 AND 7-day price trend positive
    SELL — smoothed sentiment < -0.05 AND 7-day price trend negative
    HOLD — otherwise
    """
    if df.empty or len(df) < 8:
        return "HOLD"

    recent_sent  = df["sentiment_smooth"].tail(5).mean()
    price_trend  = df["Close"].tail(7).iloc[-1] - df["Close"].tail(7).iloc[0]

    if recent_sent > 0.05 and price_trend > 0:
        return "BUY"
    elif recent_sent < -0.05 and price_trend < 0:
        return "SELL"
    return "HOLD"


# ─────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────

def backtest(df: pd.DataFrame, forward_days: int = BACKTEST_FORWARD_DAYS) -> dict | None:
    """
    For each row, generate a signal based on the rolling smoothed sentiment
    and recent price trend. Compare that signal to actual future return.
    Returns accuracy and additional stats.
    """
    if df.empty or len(df) < forward_days + 10:
        return None

    df = df.copy().reset_index(drop=True)

    # Forward return: price N days from now vs today
    df["future_return"] = df["Close"].shift(-forward_days) - df["Close"]

    # Row-wise signal using same logic as compute_signal
    df["price_trend"] = df["Close"].diff(7)
    df["signal"] = np.where(
        (df["sentiment_smooth"] > 0.05) & (df["price_trend"] > 0), "BUY",
        np.where(
            (df["sentiment_smooth"] < -0.05) & (df["price_trend"] < 0), "SELL", "HOLD"
        )
    )

    valid = df.dropna(subset=["future_return"])

    # A BUY is "correct" if future price went up; SELL correct if it went down; HOLD always neutral
    def is_correct(row):
        if row["signal"] == "BUY":
            return 1 if row["future_return"] > 0 else 0
        elif row["signal"] == "SELL":
            return 1 if row["future_return"] < 0 else 0
        else:  # HOLD — count as correct
            return 1

    valid = valid.copy()
    valid["correct"] = valid.apply(is_correct, axis=1)

    total    = len(valid)
    correct  = valid["correct"].sum()
    accuracy = correct / total * 100 if total > 0 else 0

    buy_acc = sell_acc = None
    buy_rows  = valid[valid["signal"] == "BUY"]
    sell_rows = valid[valid["signal"] == "SELL"]

    if len(buy_rows) > 0:
        buy_acc = buy_rows["correct"].mean() * 100
    if len(sell_rows) > 0:
        sell_acc = sell_rows["correct"].mean() * 100

    return {
        "accuracy":   round(accuracy, 2),
        "total_signals": total,
        "buy_accuracy":  round(buy_acc, 2)  if buy_acc  is not None else None,
        "sell_accuracy": round(sell_acc, 2) if sell_acc is not None else None,
    }


# ─────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────

def compute_correlation(df: pd.DataFrame) -> tuple[float, float]:
    """
    Returns (raw_corr, smooth_corr) between sentiment and Close price.
    """
    sub = df.dropna(subset=["sentiment", "Close"])
    raw_corr    = sub["sentiment"].corr(sub["Close"])
    smooth_corr = sub["sentiment_smooth"].corr(sub["Close"]) if "sentiment_smooth" in sub else np.nan
    return round(raw_corr, 4), round(smooth_corr, 4)


# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

DARK_TEMPLATE = "plotly_dark"

def plot_price_and_sentiment(df: pd.DataFrame, symbol: str):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["Close"],
        mode="lines", name="Close Price",
        line=dict(color="#4a9eff", width=2),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sentiment_smooth"],
        mode="lines", name=f"Sentiment ({SENTIMENT_SMOOTH_WINDOW}-day smooth)",
        line=dict(color="#ff7043", width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df["sentiment"],
        name="Raw Sentiment",
        marker_color=np.where(df["sentiment"] >= 0, "#66bb6a", "#ef5350"),
        opacity=0.35,
        yaxis="y2",
    ))

    fig.update_layout(
        title=f"{symbol} — Close Price vs Smoothed Sentiment",
        template=DARK_TEMPLATE,
        hovermode="x unified",
        yaxis=dict(title="Price (USD)", side="left"),
        yaxis2=dict(title="Sentiment Score", side="right", overlaying="y", range=[-1, 1]),
        legend=dict(orientation="h", y=-0.15),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_moving_averages(df: pd.DataFrame, symbol: str):
    df = df.copy()
    df["MA7"]  = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["Close"],  mode="lines", name="Close",  line=dict(color="#4a9eff", width=1.5)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA7"],   mode="lines", name="7-Day MA", line=dict(color="#66bb6a", dash="dash")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA30"],  mode="lines", name="30-Day MA", line=dict(color="#ffa726", dash="dot")))
    fig.update_layout(title=f"{symbol} — Price with Moving Averages", template=DARK_TEMPLATE, height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def plot_sentiment_distribution(df: pd.DataFrame):
    df = df.copy()
    df["label"] = np.select(
        [df["sentiment"] > 0.05, df["sentiment"] < -0.05],
        ["Positive", "Negative"],
        default="Neutral",
    )
    dist = df["label"].value_counts().reset_index()
    dist.columns = ["Sentiment", "Count"]

    fig = go.Figure(data=[go.Pie(
        labels=dist["Sentiment"],
        values=dist["Count"],
        hole=0.45,
        marker_colors=["#66bb6a", "#ef5350", "#ffa726"],
        hoverinfo="label+percent+value",
    )])
    fig.update_layout(title="Sentiment Distribution", template=DARK_TEMPLATE, height=360)
    st.plotly_chart(fig, use_container_width=True)


def plot_volume_vs_sentiment(df: pd.DataFrame):
    fig = go.Figure(data=go.Scatter(
        x=df["sentiment_smooth"],
        y=df["Volume"],
        mode="markers",
        marker=dict(
            size=7,
            color=df["sentiment_smooth"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Sentiment"),
        ),
        text=df["date"].astype(str),
        hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.3f}<br>Volume: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Volume vs Smoothed Sentiment",
        xaxis_title="Smoothed Sentiment",
        yaxis_title="Volume",
        template=DARK_TEMPLATE,
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_scatter(df: pd.DataFrame):
    sub = df.dropna(subset=["sentiment_smooth", "Close"])
    fig = go.Figure(data=go.Scatter(
        x=sub["sentiment_smooth"],
        y=sub["Close"],
        mode="markers",
        marker=dict(size=6, color="#4a9eff", opacity=0.6),
        text=sub["date"].astype(str),
        hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.3f}<br>Price: $%{y:.2f}<extra></extra>",
    ))

    # Trend line
    if len(sub) > 2:
        m, b = np.polyfit(sub["sentiment_smooth"], sub["Close"], 1)
        x_line = np.linspace(sub["sentiment_smooth"].min(), sub["sentiment_smooth"].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", name="Trend",
            line=dict(color="#ff7043", dash="dash"),
        ))

    fig.update_layout(
        title="Sentiment vs Closing Price (Correlation Scatter)",
        xaxis_title="Smoothed Sentiment Score",
        yaxis_title="Close Price (USD)",
        template=DARK_TEMPLATE,
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# KPI SECTION
# ─────────────────────────────────────────────

def show_kpis(df: pd.DataFrame):
    avg_sent_7  = df["sentiment_smooth"].tail(7).mean()
    avg_sent_30 = df["sentiment_smooth"].tail(30).mean()
    avg_vol_7   = df["Volume"].tail(7).mean()
    avg_vol_30  = df["Volume"].tail(30).mean()

    tail = df["Close"].dropna()
    price_change_7 = (
        (tail.iloc[-1] - tail.iloc[-7]) / tail.iloc[-7] * 100
        if len(tail) >= 7 else float("nan")
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Sentiment (7d)",  f"{avg_sent_7:.3f}")
    c2.metric("Avg Sentiment (30d)", f"{avg_sent_30:.3f}")
    c3.metric("Avg Volume (7d)",     f"{avg_vol_7:,.0f}")
    c4.metric("Avg Volume (30d)",    f"{avg_vol_30:,.0f}")
    c5.metric("7d Price Change",     f"{price_change_7:.2f}%")


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

STOCK_LIST = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]

col_left, col_right = st.columns([1, 3])
with col_left:
    selected_stock = st.selectbox("Select a Stock", STOCK_LIST)
    run = st.button("▶  Run Analysis", use_container_width=True)

if run:
    with st.spinner("Fetching data and computing analysis…"):
        df = build_combined_df(selected_stock)

    if df.empty:
        st.error("No data available for the selected stock.")
        st.stop()

    # ── Signal ──────────────────────────────────────────────────
    signal = compute_signal(df)
    css_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[signal]
    st.markdown(f"## Signal: <span class='{css_class}'>{signal}</span>", unsafe_allow_html=True)
    st.caption("Signal is based on the 5-day smoothed sentiment trend + 7-day price momentum.")

    st.divider()

    # ── KPIs ────────────────────────────────────────────────────
    st.subheader("📌 Key Metrics")
    show_kpis(df)

    # ── Correlation ─────────────────────────────────────────────
    raw_corr, smooth_corr = compute_correlation(df)
    col_a, col_b = st.columns(2)
    col_a.metric("Correlation: Raw Sentiment vs Price",    str(raw_corr),    help="Pearson r between daily raw sentiment score and closing price")
    col_b.metric("Correlation: Smoothed Sentiment vs Price", str(smooth_corr), help="Pearson r between smoothed sentiment and closing price")

    if abs(smooth_corr) >= 0.5:
        st.success(f"Strong {'positive' if smooth_corr > 0 else 'negative'} correlation ({smooth_corr}) between smoothed sentiment and price.")
    elif abs(smooth_corr) >= 0.2:
        st.info(f"Moderate correlation ({smooth_corr}) — sentiment has some predictive relationship with price.")
    else:
        st.warning(f"Weak correlation ({smooth_corr}) — sentiment alone may not be a reliable predictor for this stock.")

    st.divider()

    # ── Charts ──────────────────────────────────────────────────
    st.subheader("📊 Price & Sentiment Over Time")
    plot_price_and_sentiment(df, selected_stock)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Price with Moving Averages")
        plot_moving_averages(df, selected_stock)
    with col2:
        st.subheader("🥧 Sentiment Distribution")
        plot_sentiment_distribution(df)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📦 Volume vs Sentiment")
        plot_volume_vs_sentiment(df)
    with col4:
        st.subheader("🔗 Correlation Scatter")
        plot_correlation_scatter(df)

    st.divider()

    # ── Backtesting ─────────────────────────────────────────────
    st.subheader(f"🧪 Backtesting ({BACKTEST_FORWARD_DAYS}-Day Forward Return Accuracy)")
    bt = backtest(df, forward_days=BACKTEST_FORWARD_DAYS)
    if bt:
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Overall Accuracy",    f"{bt['accuracy']}%")
        bc2.metric("Total Signal Days",   str(bt["total_signals"]))
        bc3.metric("BUY Signal Accuracy", f"{bt['buy_accuracy']}%"  if bt["buy_accuracy"]  is not None else "N/A")
        bc4.metric("SELL Signal Accuracy",f"{bt['sell_accuracy']}%" if bt["sell_accuracy"] is not None else "N/A")
        st.caption(
            "Accuracy = % of BUY/SELL signals where the predicted direction matched the actual "
            f"{BACKTEST_FORWARD_DAYS}-day price movement. HOLD signals are counted as correct."
        )
    else:
        st.info("Not enough historical data to run a meaningful backtest.")

    st.divider()

    # ── Raw Data ────────────────────────────────────────────────
    with st.expander("🗂 View Raw Data (last 20 rows)"):
        display_cols = ["date", "Open", "High", "Low", "Close", "Volume", "sentiment", "sentiment_smooth"]
        st.dataframe(df[display_cols].tail(20).reset_index(drop=True), use_container_width=True)
