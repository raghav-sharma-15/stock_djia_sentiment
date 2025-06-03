# main.py

"""
End-to-end pipeline for Combined_News_DJIA.csv:

1) Load & melt news → ['date','label','headline'].
2) Fetch DJIA price → technical indicators.
3) Run FinBERT sentiment on every headline → attach (pos, neu, neg).
4) Aggregate daily sentiment → ['date','label','pos_avg','neu_avg','neg_avg'].
5) Merge sentiment & technicals → final CSV.
6) Train XGBoost on merged CSV → saved model.
"""

import os
import pandas as pd

from src.data_loader import load_and_melt_djia_news, build_price_dataframe
from src.sentiment_extractor import attach_sentiment_to_news
from src.feature_engineering import aggregate_daily_sentiment, compute_technicals, merge_sentiment_and_price
from src.train_model import train_xgb_on_djia


def main():
    # 1) Paths
    raw_csv = "data/Combined_News_DJIA.csv"
    merged_csv = "data/djia_merged_features.csv"
    model_out = "models/xgb_djia_sentiment_model.joblib"

    # 2) Load & melt the DJIA news
    print("1) Loading and melting DJIA news...")
    df_melted = load_and_melt_djia_news(raw_csv)

    # 3) Fetch price data for ^DJI
    print("2) Fetching DJIA historical prices…")
    df_price = build_price_dataframe(df_melted["date"])

    # 4) Compute technical indicators
    print("3) Computing technical indicators…")
    df_tech = compute_technicals(df_price)

    # 5) Run FinBERT on each headline
    print("4) Assigning sentiment scores (FinBERT)…")
    df_sent = attach_sentiment_to_news(df_melted, batch_size=32)

    # 6) Aggregate daily sentiment
    print("5) Aggregating daily sentiment…")
    df_daily_sent = aggregate_daily_sentiment(df_sent)

    # 7) Merge sentiment + technicals
    print("6) Merging sentiment with technical indicators…")
    df_merged = merge_sentiment_and_price(df_daily_sent, df_tech)

    # 8) Save merged features
    os.makedirs("data", exist_ok=True)
    df_merged.to_csv(merged_csv, index=False)
    print(f"   → Merged features saved to {merged_csv}")

    # 9) Train XGBoost on merged features
    print("7) Training XGBoost classifier…")
    train_xgb_on_djia(df_merged, model_out)

    print("\nPipeline complete! Model saved to:", model_out)


if __name__ == "__main__":
    main()