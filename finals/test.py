import requests
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
ALPHAVANTAGE_API_KEY = '51eead0d9ed64de3ac9048ffb92cdbdc'

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
        print(f"Alpha Vantage news fetch failed with status {response.status_code}")
        return pd.DataFrame(columns=["date", "sentiment"])
    try:
        data = response.json()
    except Exception as e:
        print("Failed to parse Alpha Vantage response JSON", e)
        return pd.DataFrame(columns=["date", "sentiment"])
    articles = data.get("feed", [])
    if not articles:
        print("No articles found in Alpha Vantage response")
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

def fetch_news_sentiment_combined(symbol):
    df_yf = fetch_news_sentiment_yfinance(symbol)
    df_av = fetch_news_sentiment_alphavantage(symbol)
    combined = pd.concat([df_yf, df_av])
    combined = combined.groupby("date").mean().reset_index()
    return combined

if __name__ == "__main__":
    symbol = "AAPL"  # Change to another ticker to test
    df = fetch_news_sentiment_combined(symbol)
    if df.empty:
        print(f"No sentiment data loaded for {symbol}")
    else:
        print(f"Sentiment data for {symbol}:")
        print(df.head())
        print("Columns:", df.columns.tolist())
