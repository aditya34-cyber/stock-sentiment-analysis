import streamlit as st
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Dual-Phase Stock Sentiment Analyzer", layout="wide")
st.title("📊 Dual-Phase Stock Sentiment Analyzer")
st.caption("Analyze investor sentiment trends and predict volume behavior using NewsAPI + VADER + Logistic Regression")

# -----------------------------
# User Inputs
# -----------------------------
api_key = st.text_input("🔑 Enter your NewsAPI key:", type="password", value="")
company_name = st.text_input("🏢 Enter company name (e.g., Tata Consultancy Services):", value="Tata Consultancy Services")

col1, col2 = st.columns(2)
with col1:
    early_from = st.date_input("🟢 Early Phase Start Date", pd.to_datetime("2025-10-06"))
    early_to = st.date_input("🟢 Early Phase End Date", pd.to_datetime("2025-10-20"))
with col2:
    late_from = st.date_input("🔵 Late Phase Start Date", pd.to_datetime("2025-10-21"))
    late_to = st.date_input("🔵 Late Phase End Date", pd.to_datetime("2025-11-06"))

uploaded_oct = st.file_uploader("📂 Upload October Stock Data CSV", type=["csv"])
uploaded_nov = st.file_uploader("📂 Upload November Stock Data CSV", type=["csv"])

if st.button("🚀 Run Analysis"):
    if not api_key or not uploaded_oct or not uploaded_nov:
        st.error("⚠️ Please provide API key and both stock CSV files.")
    else:
        # Initialize
        newsapi = NewsApiClient(api_key=api_key)
        analyzer = SentimentIntensityAnalyzer()

        def get_sentiment(from_date, to_date):
            articles = newsapi.get_everything(
                q=company_name,
                from_param=str(from_date),
                to=str(to_date),
                language="en",
                sort_by="relevancy",
                page_size=100
            )
            news_data = []
            for a in articles["articles"]:
                title = a["title"]
                published = pd.to_datetime(a["publishedAt"]).date()
                sentiment = analyzer.polarity_scores(title)["compound"]
                news_data.append({"Date": published, "Sentiment_Score": sentiment})
            df = pd.DataFrame(news_data)
            return df.groupby("Date")["Sentiment_Score"].mean().reset_index()

        with st.spinner("Fetching and analyzing news sentiment..."):
            sent_early = get_sentiment(early_from, early_to)
            sent_late = get_sentiment(late_from, late_to)
            sent_early["Phase"] = "Early"
            sent_late["Phase"] = "Late"
            sent_all = pd.concat([sent_early, sent_late])

        # Load Stock Data
        oct_data = pd.read_csv(uploaded_oct)
        nov_data = pd.read_csv(uploaded_nov)
        oct_data["Date"] = pd.to_datetime(oct_data["Date"]).dt.date
        nov_data["Date"] = pd.to_datetime(nov_data["Date"]).dt.date

        merged = pd.merge(oct_data, sent_all, on="Date", how="left")
        merged["Sentiment_Score"].fillna(method="ffill", inplace=True)
        merged["Sentiment_Score"].fillna(method="bfill", inplace=True)

        # Features
        merged["Volume_Change_%"] = merged["Volume"].pct_change() * 100
        merged["Vol_MA3"] = merged["Volume"].rolling(3).mean()
        merged["Vol_Trend"] = (merged["Volume"] > merged["Vol_MA3"]).astype(int)
        merged["Sentiment_Lag1"] = merged["Sentiment_Score"].shift(1)
        merged["Weighted_Sentiment"] = merged["Sentiment_Score"] * merged["Volume_Change_%"]

        avg_early = sent_early["Sentiment_Score"].mean()
        avg_late = sent_late["Sentiment_Score"].mean()
        merged["Sentiment_Shift"] = avg_late - avg_early

        merged["Next_Volume"] = merged["Volume"].shift(-1)
        merged["Volume_Up"] = np.where(merged["Next_Volume"] > merged["Volume"], 1, 0)
        merged = merged.dropna()

        # Model Training
        features = [
            "Sentiment_Score", "Sentiment_Lag1",
            "Weighted_Sentiment", "Volume_Change_%",
            "Vol_Trend", "Sentiment_Shift"
        ]

        X = merged[features]
        y = merged["Volume_Up"]

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        merged["Predicted_Up"] = model.predict(X)
        accuracy = accuracy_score(y, merged["Predicted_Up"])

        st.success(f"📈 Model Accuracy: **{accuracy*100:.2f}%**")

        # Advice
        if avg_early > 0 and avg_late > avg_early:
            advice = "📈 Sentiment improving — buyers gaining confidence. Strong BUY signal."
        elif avg_early > 0 and avg_late < avg_early:
            advice = "⚖️ Sentiment weakening — consider partial profit booking."
        elif avg_early < 0 and avg_late > 0:
            advice = "📉 Sentiment reversal — possible recovery, watch closely."
        else:
            advice = "🚫 Consistently negative sentiment — avoid new entries."

        st.subheader("💬 Investment Advice")
        st.info(advice)

        # Visualization
        st.subheader("📊 Sentiment Evolution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sent_early["Date"], sent_early["Sentiment_Score"], label="Early Phase (T1)", marker="o")
        ax.plot(sent_late["Date"], sent_late["Sentiment_Score"], label="Late Phase (T2)", marker="x")
        ax.set_title(f"{company_name} Sentiment Evolution ({early_from}–{late_to})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Save results
        csv = merged.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Full Results CSV", data=csv, file_name=f"{company_name.replace(' ', '_')}_Results.csv")

        st.balloons()