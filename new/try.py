import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1 — Initialize
# -----------------------------
API_KEY = "7cdc5078f01a4f67a4d0ca9258137bb1" # Replace with your actual key
company_name = "Tata Consultancy Services" 

newsapi = NewsApiClient(api_key=API_KEY)
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(from_date, to_date):
    """Fetch average daily sentiment for a date range."""
    articles = newsapi.get_everything(
        q=company_name,
        from_param=from_date,
        to=to_date,
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

# -----------------------------
# STEP 2 — Get Dual Sentiment Phases
# -----------------------------
sent_early = get_sentiment("2025-10-06", "2025-10-20")
sent_late = get_sentiment("2025-10-21", "2025-11-06")

sent_early["Phase"] = "Early"
sent_late["Phase"] = "Late"
sent_all = pd.concat([sent_early, sent_late])

# -----------------------------
# STEP 3 — Load Stock Data (TCS October + November)
# -----------------------------
oct_data = pd.read_csv("oct_2025.csv")
nov_data = pd.read_csv("nov_2025.csv")

oct_data["Date"] = pd.to_datetime(oct_data["Date"]).dt.date
nov_data["Date"] = pd.to_datetime(nov_data["Date"]).dt.date

# -----------------------------
# STEP 4 — Merge Sentiment with Stock Volume
# -----------------------------
merged = pd.merge(oct_data, sent_all, on="Date", how="left")
merged["Sentiment_Score"].fillna(method="ffill", inplace=True)
merged["Sentiment_Score"].fillna(method="bfill", inplace=True)

# -----------------------------
# STEP 5 — Add Technical & Sentiment Features
# -----------------------------
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

# -----------------------------
# STEP 6 — Train Logistic Regression Model
# -----------------------------
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

print(f"\n📊 Dual-Phase Hybrid Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# STEP 7 — Integrated Investment Advice
# -----------------------------
if avg_early > 0 and avg_late > avg_early:
    advice = "📈 Sentiment improving — buyers gaining confidence. Strong BUY signal."
elif avg_early > 0 and avg_late < avg_early:
    advice = "⚖️ Sentiment weakening — consider partial profit booking."
elif avg_early < 0 and avg_late > 0:
    advice = "📉 Sentiment reversal — possible recovery, watch closely."
else:
    advice = "🚫 Consistently negative sentiment — avoid new entries."

print("\n💬 Integrated Sentiment Advice:")
print(advice)

# -----------------------------
# STEP 8 — Visualization
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(sent_early["Date"], sent_early["Sentiment_Score"], label="Early Phase (T1)", marker="o")
plt.plot(sent_late["Date"], sent_late["Sentiment_Score"], label="Late Phase (T2)", marker="x")
plt.title(f"{company_name} Sentiment Evolution (Oct–Nov 2025)")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 9 — Save Results
# -----------------------------
merged.to_csv(f"{company_name.replace(' ', '_')}_Dual_Hybrid_Sentiment_Analysis_Oct2025.csv", index=False)
print(f"\n✅ Saved results to '{company_name.replace(' ', '_')}_Dual_Hybrid_Sentiment_Analysis_Oct2025.csv'")