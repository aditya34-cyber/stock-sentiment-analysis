import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1 — Load Stock Data (October + November)
# -----------------------------
oct_data = pd.read_csv("oct_2025.csv")
nov_data = pd.read_csv("nov_2025.csv")

oct_data["Date"] = pd.to_datetime(oct_data["Date"])
nov_data["Date"] = pd.to_datetime(nov_data["Date"])

# -----------------------------
# STEP 2 — Fetch News from NewsAPI (October 2025)
# -----------------------------
API_KEY = "7cdc5078f01a4f67a4d0ca9258137bb1"  # Replace locally with your real key
company_name = "Tata Consultancy Services"  # Change as needed (TCS / Reliance)

newsapi = NewsApiClient(api_key=API_KEY)

articles = newsapi.get_everything(
    q=company_name,
    from_param="2025-10-10",
    to="2025-10-31",
    language="en",
    sort_by="relevancy",
    page_size=100
)

print(f"📰 Fetched {len(articles['articles'])} news articles for {company_name} (October 2025)")

# -----------------------------
# STEP 3 — Sentiment Analysis (VADER)
# -----------------------------
analyzer = SentimentIntensityAnalyzer()
news_data = []

for a in articles["articles"]:
    title = a["title"]
    published = pd.to_datetime(a["publishedAt"]).date()
    sentiment = analyzer.polarity_scores(title)["compound"]
    news_data.append({"Date": published, "Headline": title, "Sentiment_Score": sentiment})

news_df = pd.DataFrame(news_data)
daily_sentiment = news_df.groupby("Date")["Sentiment_Score"].mean().reset_index()

# -----------------------------
# STEP 4 — Merge Sentiment with Stock Volume
# -----------------------------
oct_data["Date"] = oct_data["Date"].dt.date
daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"]).dt.date

merged = pd.merge(oct_data, daily_sentiment, on="Date", how="left")
merged["Sentiment_Score"].fillna(0, inplace=True)

# -----------------------------
# STEP 5 — Add Technical Features
# -----------------------------
merged["Volume_Change_%"] = merged["Volume"].pct_change() * 100
merged["Vol_MA3"] = merged["Volume"].rolling(3).mean()
merged["Vol_Trend"] = (merged["Volume"] > merged["Vol_MA3"]).astype(int)
merged["Sentiment_Lag1"] = merged["Sentiment_Score"].shift(1)
merged["Weighted_Sentiment"] = merged["Sentiment_Score"] * merged["Volume_Change_%"]

merged["Next_Volume"] = merged["Volume"].shift(-1)
merged["Volume_Up"] = np.where(merged["Next_Volume"] > merged["Volume"], 1, 0)
merged = merged.dropna()

# -----------------------------
# STEP 6 — Train Logistic Regression Model
# -----------------------------
features = [
    "Sentiment_Score", "Sentiment_Lag1",
    "Weighted_Sentiment", "Volume_Change_%", "Vol_Trend"
]

X = merged[features]
y = merged["Volume_Up"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

merged["Predicted_Up"] = model.predict(X)
accuracy = accuracy_score(y, merged["Predicted_Up"])

print(f"\n📊 Hybrid Model Accuracy (October 2025): {accuracy*100:.2f}%")

# -----------------------------
# STEP 7 — Investment Advice for November
# -----------------------------
avg_sentiment = merged["Sentiment_Score"].mean()
avg_weighted = merged["Weighted_Sentiment"].mean()
avg_volume_trend = merged["Vol_Trend"].mean()

if avg_sentiment > 0.05 and avg_weighted > 0 and avg_volume_trend > 0.5:
    advice = "📈 Strong sentiment & high volume momentum — likely BUY signal for November."
elif avg_sentiment < -0.05 and avg_weighted < 0:
    advice = "📉 Negative sentiment & weak participation — AVOID new investments."
else:
    advice = "⚖️ Mixed signals — HOLD existing positions."

print("\n💬 Investment Advice for November 2025:")
print(advice)

# -----------------------------
# STEP 8 — Visualization
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(oct_data["Date"], oct_data["Volume"], label="October Volume", marker="o")
plt.plot(nov_data["Date"], nov_data["Volume"], label="November Volume", marker="x")
plt.title(f"{company_name} — October vs November 2025 Volume Trend")
plt.xlabel("Date")
plt.ylabel("Volume (Shares)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 9 — Save Results
# -----------------------------
merged.to_csv(f"{company_name.replace(' ', '_')}_Hybrid_Sentiment_Analysis_Oct2025.csv", index=False)
print(f"\n✅ Saved results to '{company_name.replace(' ', '_')}_Hybrid_Sentiment_Analysis_Oct2025.csv'")