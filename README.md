

📊 Dual-Phase Stock Sentiment Analyzer

News-Driven AI for Predicting Stock Market Behavior

This project is an AI-powered web app that analyzes investor sentiment trends over time using live news data and machine learning.
It evaluates the change in sentiment between two time phases (Early and Late) to predict stock volume movement and provide actionable investment advice.

⸻

🚀 Features

✅ Dual-Phase Sentiment Analysis
Splits a date range into two phases to detect shifts in market mood over time.

✅ NewsAPI Integration
Fetches company-specific financial news directly from the web.

✅ VADER Sentiment Engine
Performs natural language sentiment scoring on each news headline.

✅ Hybrid ML Model
Combines sentiment features with technical stock indicators using Logistic Regression.

✅ Interactive Web Interface (Streamlit)
Upload stock data, view sentiment charts, get BUY/HOLD/SELL advice, and download full results.

✅ Real-Time Insights
See how emotional tone in the market correlates with actual stock behavior.

⸻

🧠 How It Works

The app combines news sentiment and stock trading activity to predict market direction.
It works in the following stages:

Step	Process	Description
1️⃣	Fetch News Data	Pulls live articles for the selected company and date range using NewsAPI.
2️⃣	Sentiment Analysis	Uses VADER to calculate a Sentiment_Score for each article.
3️⃣	Dual Phase Segmentation	Splits data into Early Phase and Late Phase to capture sentiment drift.
4️⃣	Feature Engineering	Creates hybrid features like Volume_Change_%, Vol_Trend, Weighted_Sentiment, and Sentiment_Shift.
5️⃣	Model Training	Logistic Regression model learns to predict whether stock volume will rise or fall.
6️⃣	Investment Advice	The app evaluates sentiment changes and recommends: BUY, HOLD, or AVOID.
7️⃣	Visualization & Export	Displays interactive charts and allows CSV download of results.


⸻

🧩 Tech Stack

Component	Technology
Frontend / UI	Streamlit
Backend / Logic	Python 3.10+
APIs	NewsAPI (https://newsapi.org)
NLP Engine	VADER Sentiment Analyzer
Machine Learning	Logistic Regression (Scikit-Learn)
Visualization	Matplotlib
Data Handling	Pandas, NumPy


⸻

⚙️ Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/yourusername/dual-phase-stock-sentiment.git
cd dual-phase-stock-sentiment

2️⃣ Install dependencies

pip install streamlit pandas numpy newsapi-python vaderSentiment scikit-learn matplotlib

3️⃣ Run the web app

streamlit run app.py

4️⃣ Open in browser

Navigate to:
👉 http://localhost:8501￼

⸻

🧾 Input Requirements

📂 Stock Data CSVs

You must upload two CSV files:
	•	oct_2025.csv
	•	nov_2025.csv

Each file should contain:

Column	Description
Date	Date of trading session
Volume	Number of shares traded

Example:

Date,Volume
2025-10-06,2112774
2025-10-07,3062943
...

🔑 NewsAPI Key

You need a free API key from https://newsapi.org￼.

⸻

📈 Outputs

After running the analysis, the app displays:

Output	Description
Model Accuracy	Predictive performance (typically ~80%)
Sentiment Graphs	Early vs Late phase sentiment evolution
Investment Advice	BUY / HOLD / AVOID recommendation
Downloadable CSV	Detailed data with all features and predictions


⸻

🧠 Example Interpretation

If the early phase sentiment was +0.12 and late phase sentiment improved to +0.25,
→ The Sentiment_Shift = +0.13 → indicating increasing optimism
✅ The model might predict rising volume → BUY signal.

If sentiment worsens (negative shift),
⚠️ It suggests caution or partial profit booking.

⸻

📊 Visualization Example

The app generates an interactive chart showing:

Sentiment Score
│
│        🔼 Early Phase (T1)
│     ___/
│    /
│   /           🔽 Late Phase (T2)
│__/______________________________→ Time

This helps visualize whether market mood is strengthening or fading.

⸻

💬 Investment Advice Logic

Condition	Sentiment Trend	Advice
avg_early < avg_late	Improving	📈 BUY / STRONG BULLISH
avg_early > avg_late	Weakening	⚖️ HOLD / PARTIAL EXIT
Both Negative	Bearish	🚫 AVOID ENTRY
Both Positive	Stable	💹 HOLD CONFIDENTLY


⸻

🧮 Example Output (for TCS)

📊 Dual-Phase Hybrid Model Accuracy: 80.00%
💬 Integrated Sentiment Advice:
📈 Sentiment improving — buyers gaining confidence. Strong BUY signal.


⸻

🧑‍💻 Developer Info

Developer: Aditya Kulkarni
Tech: Python | NLP | Machine Learning | Web Development
Focus: Stock Market Sentiment Analysis
GitHub: github.com/yourusername￼

⸻

🛠️ Future Enhancements
	•	Add word clouds for positive vs negative headlines
	•	Integrate price prediction alongside volume prediction
	•	Support multiple companies at once
	•	Deploy on Streamlit Cloud / Hugging Face Spaces
	•	Add a dark theme + dashboard UI

⸻

⚖️ Disclaimer

This app is built for educational and research purposes only.
It should not be used for actual trading or financial advice.
Stock markets involve risk — use at your own discretion.

⸻

🌟 Acknowledgments
	•	NewsAPI￼ for news data
	•	VADER Sentiment￼ for NLP analysis
	•	Scikit-Learn￼ for ML modeling
	•	Streamlit￼ for interactive deployment

⸻

💥 “Predicting sentiment is easy. Understanding sentiment shifts is power.”

— Aditya Kulkarni, 2025

