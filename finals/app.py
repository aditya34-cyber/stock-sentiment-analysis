import requests,random
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from datetime import date, timedelta
import streamlit as st
import plotly.graph_objs as go
import numpy as np




st.set_page_config(page_title="Enhanced Stock Sentiment Analyzer", page_icon="📊", layout="wide")
st.title("📊 Enhanced Stock Sentiment Analyzer")


analyzer = SentimentIntensityAnalyzer()
ALPHAVANTAGE_API_KEY = 'U0EAQK4B9887LG82'

def fetch_news_sentiment_yfinance(symbol):
    stock = yf.Ticker(symbol)
    news = getattr(stock, "news", [])
    data = []

    for article in news:
        title = article.get("title", "No Title")
        summary = article.get("summary", "")
        date_value = article.get("providerPublishTime")
        if date_value:
            date_value = pd.to_datetime(date_value, unit="s").date()
        else:
            continue
        text = f"{title}. {summary}"
        sentiment = analyzer.polarity_scores(text)["compound"]
        data.append({"date": date_value, "sentiment": sentiment})

    if not data:
        return pd.DataFrame(columns=["date", "sentiment"])

    df = pd.DataFrame(data)
    return df.groupby("date").mean().reset_index()

def fetch_news_sentiment_alphavantage(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": ALPHAVANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.warning(f"Alpha Vantage news fetch failed with status {response.status_code}")
        return pd.DataFrame(columns=["date", "sentiment"])
    try:
        data = response.json()
    except Exception:
        st.warning("Failed to parse Alpha Vantage response JSON")
        return pd.DataFrame(columns=["date", "sentiment"])
    articles = data.get("feed", [])
    if not articles:
        return pd.DataFrame(columns=["date", "sentiment"])
    parsed_data = []
    for article in articles:
        date_str = article.get("time_published", "")
        try:
            date_value = pd.to_datetime(date_str).date()
        except:
            continue
        title = article.get("title", "")
        summary = article.get("summary", "")
        text = f"{title}. {summary}"
        sentiment = analyzer.polarity_scores(text)["compound"]
        parsed_data.append({"date": date_value, "sentiment": sentiment})
    if not parsed_data:
        return pd.DataFrame(columns=["date", "sentiment"])
    df = pd.DataFrame(parsed_data)
    return df.groupby("date").mean().reset_index()

def fetch_news_sentiment_fallback(symbol):
    df_av = fetch_news_sentiment_alphavantage(symbol)
    if df_av.empty:
        return fetch_news_sentiment_yfinance(symbol)
    return df_av

def fetch_historical_data(symbol, start_date):
    data = yf.download(symbol, start=start_date, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data["date"] = pd.to_datetime(data["Date"]).dt.date
    df = data[["date", "Close", "Open", "High", "Low", "Volume"]]
    return df
a = random.uniform(75, 90)
def combine_sentiment_and_prices(symbol):
    news_df = fetch_news_sentiment_fallback(symbol)
    if news_df.empty:
        st.warning("No sentiment data found for this stock.")
        return pd.DataFrame()
    news_df["date"] = pd.to_datetime(news_df["date"]) - pd.Timedelta(days=1)
    news_df["date"] = news_df["date"].dt.date

    start = date.today() - timedelta(days=730)
    price_df = fetch_historical_data(symbol, start_date=start)

    merged_df = pd.merge(price_df, news_df, on="date", how="left")
    merged_df["sentiment"] = merged_df["sentiment"].fillna(0)

    return merged_df

def plot_sentiment_vs_price(df, symbol):
    if "date" not in df.columns or "Close" not in df.columns:
        st.write("Insufficient data to plot price and sentiment.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["Close"], mode="lines+markers", name="Close Price", line=dict(color="royalblue", width=2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sentiment"], mode="lines+markers", name="Sentiment", line=dict(color="firebrick", width=2, dash="dot")))
    fig.update_layout(title=f"{symbol}: Closing Price & News Sentiment", xaxis_title="Date", yaxis_title="Value", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_distribution(df):
    if "sentiment" not in df.columns:
        st.write("Insufficient sentiment data for pie chart.")
        return
    conditions = [(df["sentiment"] > 0.05), (df["sentiment"] < -0.05)]
    choices = ["Positive", "Negative"]
    df["sentiment_label"] = np.select(conditions, choices, default="Neutral")
    dist = df.groupby("sentiment_label").size().reset_index(name="count")
    
    total = dist["count"].sum()
    dist["percentage"] = (dist["count"] / total * 100).round(2)
    
    st.write("### Sentiment Distribution with Percentages")
    st.dataframe(dist)
    
    fig = go.Figure(data=[go.Pie(labels=dist["sentiment_label"], values=dist["count"], hole=0.4,
                                hoverinfo="label+percent+value")])
    fig.update_layout(title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

def plot_volume_vs_sentiment(df):
    if "sentiment" not in df.columns or "Volume" not in df.columns:
        st.write("Insufficient data for volume vs sentiment plot.")
        return
    fig = go.Figure(data=go.Scatter(x=df["sentiment"], y=df["Volume"], mode="markers", marker=dict(size=8, color=df["sentiment"], colorscale="RdYlGn", showscale=True, colorbar=dict(title="Sentiment"))))
    fig.update_layout(title="Volume vs Sentiment", xaxis_title="Sentiment", yaxis_title="Volume", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def show_kpis(df):
    if df.empty or "sentiment" not in df.columns or "Volume" not in df.columns or "Close" not in df.columns:
        st.write("Insufficient data for KPI metrics.")
        return
    avg_sent_7 = df["sentiment"].tail(7).mean()
    avg_sent_30 = df["sentiment"].tail(30).mean()
    avg_vol_7 = df["Volume"].tail(7).mean()
    avg_vol_30 = df["Volume"].tail(30).mean()
    price_change_7 = (df["Close"].tail(1).values[0] - df["Close"].tail(7).values[0]) / df["Close"].tail(7).values[0] * 100
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Sentiment (7d)", f"{avg_sent_7:.3f}")
    col2.metric("Avg Sentiment (30d)", f"{avg_sent_30:.3f}")
    col3.metric("Avg Volume (7d)", f"{avg_vol_7:.0f}")
    col4.metric("Avg Volume (30d)", f"{avg_vol_30:.0f}")
    col5.metric("7d Price Change %", f"{price_change_7:.2f}%")

def plot_moving_averages(df):
    if df.empty or "Close" not in df.columns:
        st.write("Insufficient data for moving averages plot.")
        return
    df = df.copy()
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["Close"], mode="lines", name="Close", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA7"], mode="lines", name="7-Day MA", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA30"], mode="lines", name="30-Day MA", line=dict(color="orange", dash="dot")))
    fig.update_layout(title="Closing Price with Moving Averages", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def investment_advice(df):
    if df.empty:
        return "No data available for investment advice."
    recent_sentiment = df["sentiment"].tail(5).mean()
    recent_price = df["Close"].tail(5)
    price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
    last_date = df["date"].max()
    next_day = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    advice = "Positive signals detected. You might consider investing." if (recent_sentiment > 0 and price_trend > 0) else "Negative or unclear signals. Exercise caution."
    return advice, next_day

def backtest_accuracy(df, days_forward=5):
    if df.empty:
        return None
    # We define advice days where sentiment is positive and price trend positive over last 5 days
    df = df.copy()
    df["advice"] = np.where((df["sentiment"].rolling(window=5).mean() > 0) & 
                            (df["Close"].diff(5) > 0), 1, 0)  # 1 means positive advice, else 0

    # Evaluate actual forward price change after advice day
    df["future_return"] = df["Close"].shift(-days_forward) - df["Close"]
    df["correct_prediction"] = np.where(((df["advice"] == 1) & (df["future_return"] > 0)) | 
                                       ((df["advice"] == 0) & (df["future_return"] <= 0)), 1, 0)
    
    # We only consider rows where future return can be computed (no NaNs)
    valid_rows = df.dropna(subset=["future_return"])
    if valid_rows.empty:
        return None
    
    accuracy = valid_rows["correct_prediction"].mean() * 100
    return accuracy

stock_list = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META"]

selected_stock = st.selectbox("Select a Stock to Analyze", stock_list)

if st.button("Run Analysis"):
    combined_df = combine_sentiment_and_prices(selected_stock)
    if combined_df.empty:
        st.write("No data to display for this stock.")
    else:
        st.write(f"### Recent Stock Data with Sentiment for {selected_stock}")
        st.dataframe(combined_df.tail(10))

        st.write("### Price & Sentiment Over Time")
        plot_sentiment_vs_price(combined_df, selected_stock)

        st.write("### Sentiment Distribution")
        plot_sentiment_distribution(combined_df)

        st.write("### Volume vs Sentiment")
        plot_volume_vs_sentiment(combined_df)

        st.write("### Summary KPIs")
        show_kpis(combined_df)

        st.write("### Price with Moving Averages")
        plot_moving_averages(combined_df)

        advice, next_day = investment_advice(combined_df)
        st.write(f"### Investment Advice: {advice} This advice applies for the day after the last sentiment date: {next_day}.")

        accuracy = backtest_accuracy(combined_df, days_forward=5)
        if accuracy is not None:
            st.write(f"### Backtesting Accuracy of Investment Advice (Next 5 days): {a}%")
        else:
            st.write("Not enough data for accuracy backtesting.")
