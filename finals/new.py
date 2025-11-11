import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import date, timedelta
import streamlit as st
import plotly.graph_objs as go
import numpy as np

st.set_page_config(page_title="Enhanced Stock Sentiment Analyzer", page_icon="📊", layout="wide")
st.title("📊 Enhanced Stock Sentiment Analyzer")

analyzer = SentimentIntensityAnalyzer()


def fetch_news_sentiment(symbol):
    stock = yf.Ticker(symbol)
    news = stock.news

    data = []
    for article in news:
        content = article.get('content', {})
        title = content.get('title', 'No Title')
        summary = content.get('summary', '')

        date_value = article.get('providerPublishTime', None)
        if not date_value:
            pubDate = content.get('pubDate', None)
            displayTime = content.get('displayTime', None)
            date_str = pubDate or displayTime
            if date_str:
                try:
                    date_value = pd.to_datetime(date_str).date()
                except Exception:
                    date_value = None
            else:
                date_value = None
        else:
            date_value = pd.to_datetime(date_value, unit='s').date()

        if not date_value:
            continue

        text_to_analyze = f"{title}. {summary}"
        sentiment = analyzer.polarity_scores(text_to_analyze)['compound']
        data.append({'date': date_value, 'sentiment': sentiment})

    if not data:
        return pd.DataFrame(columns=['date', 'sentiment'])

    df = pd.DataFrame(data)
    daily_sentiment = df.groupby('date').mean().reset_index()
    return daily_sentiment


def fetch_historical_data(symbol, start_date=None):
    stock = yf.Ticker(symbol)
    if start_date:
        hist = stock.history(start=start_date, end=pd.Timestamp.today() + pd.Timedelta(days=1))
    else:
        hist = stock.history(period="2y", end=pd.Timestamp.today() + pd.Timedelta(days=1))
    hist.reset_index(inplace=True)
    hist['date'] = hist['Date'].dt.date
    return hist[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]


def combine_sentiment_and_prices(symbol):
    news_df = fetch_news_sentiment(symbol)
    news_df['date'] = pd.to_datetime(news_df['date']) - pd.Timedelta(days=1)
    news_df['date'] = news_df['date'].dt.date

    start = date.today() - timedelta(days=730)
    price_df = fetch_historical_data(symbol, start_date=start)

    merged_df = pd.merge(price_df, news_df, on='date', how='left')
    merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
    return merged_df


def plot_sentiment_vs_price(df, symbol):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['Close'],
        mode='lines+markers',
        name='Closing Price',
        line=dict(color='royalblue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['sentiment'],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='firebrick', width=2, dash='dot')
    ))

    fig.update_layout(
        title=f"{symbol}: Closing Price & News Sentiment",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def investment_advice(df):
    recent_sentiment = df['sentiment'].tail(5).mean()
    recent_price = df['Close'].tail(5)
    price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
    if recent_sentiment > 0 and price_trend > 0:
        return "Positive signals detected. You might consider investing."
    else:
        return "Signals unclear or negative. Exercise caution or seek further analysis."


def plot_sentiment_distribution(df):
    # Categorize sentiments into pos, neu, neg
    conditions = [
        (df['sentiment'] > 0.05),
        (df['sentiment'] < -0.05)
    ]
    choices = ['Positive', 'Negative']
    df['sentiment_label'] = np.select(conditions, choices, default='Neutral')

    dist = df.groupby('sentiment_label').size().reset_index(name='count')
    fig = go.Figure(data=[go.Pie(labels=dist['sentiment_label'], values=dist['count'], hole=0.4)])
    fig.update_layout(title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)


def plot_volume_vs_sentiment(df):
    fig = go.Figure(data=go.Scatter(
        x=df['sentiment'],
        y=df['Volume'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['sentiment'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Sentiment')
        )
    ))
    fig.update_layout(title='Volume vs Sentiment', xaxis_title='Sentiment', yaxis_title='Volume', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def show_kpis(df):
    avg_sent_7 = df['sentiment'].tail(7).mean()
    avg_sent_30 = df['sentiment'].tail(30).mean()
    avg_vol_7 = df['Volume'].tail(7).mean()
    avg_vol_30 = df['Volume'].tail(30).mean()
    price_change_7 = (df['Close'].tail(1).values[0] - df['Close'].tail(7).values[0]) / df['Close'].tail(7).values[0] * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Sentiment (7d)", f"{avg_sent_7:.3f}")
    col2.metric("Avg Sentiment (30d)", f"{avg_sent_30:.3f}")
    col3.metric("Avg Volume (7d)", f"{avg_vol_7:.0f}")
    col4.metric("Avg Volume (30d)", f"{avg_vol_30:.0f}")
    col5.metric("7d Price Change %", f"{price_change_7:.2f}%")

def plot_moving_averages(df):
    df = df.copy()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['Close'],
        mode='lines',
        name='Close',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA7'],
        mode='lines',
        name='7-Day MA',
        line=dict(color='green', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA30'],
        mode='lines',
        name='30-Day MA',
        line=dict(color='orange', dash='dot')
    ))

    fig.update_layout(
        title='Closing Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


stock_list = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META']
selected_stock = st.selectbox("Select a Stock to Analyze", stock_list)

if st.button("Run Analysis"):
    combined_df = combine_sentiment_and_prices(selected_stock)

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

    advice = investment_advice(combined_df)
    st.write(f"### Investment Advice: {advice}")
