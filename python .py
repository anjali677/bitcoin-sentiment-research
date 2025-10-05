# ==============================================================
# Project: Bitcoin Market Sentiment vs Trader Performance
# Author: Vaishali
# ==============================================================

# === 1. Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 2. Load Datasets ===
trader_path = r"C:\Users\hp\New folder\python pratice\historical_trader_data.csv"
sentiment_path = r"C:\Users\hp\New folder\python pratice\fear_greed_index.csv"

trader_df = pd.read_csv(trader_path)
sentiment_df = pd.read_csv(sentiment_path)

print("✅ Datasets Loaded Successfully!")
print(f"Trader Data: {trader_df.shape}")
print(f"Sentiment Data: {sentiment_df.shape}\n")

print("Trader Columns:", list(trader_df.columns))
print("Sentiment Columns:", list(sentiment_df.columns))

# === 3. Auto-detect time column in trader data ===
time_cols = [c for c in trader_df.columns if "time" in c.lower() or "timestamp" in c.lower() or "date" in c.lower()]
if not time_cols:
    raise KeyError("❌ No time-related column found in trader data.")
else:
    time_col = time_cols[0]
    print(f"⏰ Using '{time_col}' as trader time column.")
    trader_df[time_col] = pd.to_datetime(trader_df[time_col], errors='coerce')
    trader_df = trader_df.dropna(subset=[time_col])
    trader_df = trader_df.sort_values(time_col)

# === 4. Auto-detect date column in sentiment data ===
sentiment_date_cols = [c for c in sentiment_df.columns if "date" in c.lower() or "time" in c.lower()]
if not sentiment_date_cols:
    raise KeyError("❌ No date or time column found in sentiment data. Check your CSV header names.")
else:
    sentiment_date_col = sentiment_date_cols[0]
    print(f"📅 Using '{sentiment_date_col}' as sentiment date column.")
    sentiment_df[sentiment_date_col] = pd.to_datetime(sentiment_df[sentiment_date_col], errors='coerce')
    sentiment_df = sentiment_df.dropna(subset=[sentiment_date_col])
    sentiment_df = sentiment_df.sort_values(sentiment_date_col)

# === 5. Merge trader data with nearest sentiment date ===
merged_df = pd.merge_asof(
    trader_df,
    sentiment_df,
    left_on=time_col,
    right_on=sentiment_date_col,
    direction='backward'
)

print("\n✅ Merged Dataset Created!")
print(merged_df.head())

# === 6. Basic EDA ===
sent_col = None
for c in merged_df.columns:
    if 'classification' in c.lower() or 'sentiment' in c.lower():
        sent_col = c
        break

if sent_col is None:
    raise KeyError("❌ Could not find 'Classification' or 'Sentiment' column in sentiment data.")

print("\n=== Sentiment Distribution ===")
print(merged_df[sent_col].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x=sent_col, data=merged_df, palette='coolwarm')
plt.title("Market Sentiment Distribution (Fear vs Greed)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# === 7. Trader Performance Analysis ===
print("\n=== Trader Performance Summary ===")
if 'Closed PnL' in merged_df.columns:
    pnl_col = 'Closed PnL'
else:
    pnl_col = [c for c in merged_df.columns if 'pnl' in c.lower()][0]

performance_summary = merged_df.groupby(sent_col)[pnl_col].describe()
print(performance_summary)

plt.figure(figsize=(8,5))
sns.boxplot(x=sent_col, y=pnl_col, data=merged_df, palette='Set2')
plt.title("Trader Profit/Loss by Market Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Closed PnL")
plt.show()

# === 8. Leverage and Risk Behavior ===
if 'leverage' in merged_df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=sent_col, y='leverage', data=merged_df, palette='Set1')
    plt.title("Leverage Behavior under Different Sentiments")
    plt.xlabel("Sentiment")
    plt.ylabel("Leverage")
    plt.show()

    avg_stats = merged_df.groupby(sent_col)[['leverage', pnl_col]].mean()
    print("\n=== Average Leverage and PnL by Sentiment ===")
    print(avg_stats)

# === 9. Trade Side Analysis (Long vs Short) ===
side_col = [c for c in merged_df.columns if 'side' in c.lower()]
if side_col:
    side_col = side_col[0]
    plt.figure(figsize=(7,5))
    sns.countplot(x=sent_col, hue=side_col, data=merged_df)
    plt.title("Trade Side (Long/Short) vs Market Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Trades")
    plt.legend(title='Trade Side')
    plt.show()

# === 10. Correlation Analysis ===
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
corr_df = merged_df[numeric_cols].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr_df, annot=True, cmap='Blues')
plt.title("Correlation Heatmap of Key Trading Metrics")
plt.show()

# === 11. Sentiment-wise Profitability Trend Over Time ===
plt.figure(figsize=(10,5))
merged_df.groupby([sentiment_date_col, sent_col])[pnl_col].mean().unstack().plot(kind='line')
plt.title("Average Trader PnL Over Time by Sentiment")
plt.xlabel("Date")
plt.ylabel("Average Closed PnL")
plt.legend(title="Sentiment")
plt.show()

# === 12. Save Cleaned & Merged Data ===
merged_df.to_csv("merged_trader_sentiment_data.csv", index=False)
print("\n✅ Analysis Complete! Cleaned dataset saved as 'merged_trader_sentiment_data.csv'.")
