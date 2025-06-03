# src/feature_engineering.py

import pandas as pd


def aggregate_daily_sentiment(df_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df_sent with columns ['date','label','headline','pos','neu','neg'].
    1) Group by date, average (pos, neu, neg).
    2) Keep the original 'label' (0/1) once per date.
    Returns: DataFrame with columns ['date','label','pos_avg','neu_avg','neg_avg'].
    """
    # 1) Average sentiment per date
    df_agg_sent = (
        df_sent.groupby("date", as_index=False)[["pos", "neu", "neg"]]
        .mean()
        .rename(columns={"pos": "pos_avg", "neu": "neu_avg", "neg": "neg_avg"})
    )

    # 2) Extract one 'label' per date (all headlines of a day share the same label)
    df_label = df_sent[["date", "label"]].drop_duplicates(subset=["date"]).reset_index(drop=True)

    # 3) Merge label + aggregated scores
    df_daily_sent = pd.merge(df_label, df_agg_sent, how="left", on="date")
    return df_daily_sent  # columns: date, label, pos_avg, neu_avg, neg_avg



def compute_technicals(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame df_price with columns ['date','Open','High','Low','Close','Volume'],
    this computes:
      - Return: (Close / previous Close) - 1
      - EMA_10: 10-day exponential moving average on Close
      - SMA_20: 20-day simple moving average on Close
      - RSI_14: 14-day Relative Strength Index on Close
      - ATR_14: 14-day Average True Range using High, Low, Close
      - Volume_lag1: yesterday's Volume

    Returns a DataFrame with all original price columns plus:
      ['Return','EMA_10','SMA_20','RSI_14','ATR_14','Volume_lag1'].
    Drops initial rows where any of these are NaN.
    """

    # 1) Verify the input has the required columns
    required = {"date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df_price.columns)
    if missing:
        raise KeyError(
            f"compute_technicals() expected columns {required}, but found only {set(df_price.columns)}"
        )

    # 2) Sort by date and work on a copy
    df = df_price.sort_values("date").reset_index(drop=True).copy()

    # -------------------------------
    # 3) Daily Return
    # -------------------------------
    df["Return"] = df["Close"].pct_change()

    # -------------------------------
    # 4) EMA(10) on Close
    # -------------------------------
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # -------------------------------
    # 5) SMA(20) on Close
    # -------------------------------
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # -------------------------------
    # 6) RSI(14) on Close
    # -------------------------------
    # a) Differences
    delta = df["Close"].diff()

    # b) Separate gains & losses
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # c) Simple rolling mean over 14 periods
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    # d) RS & RSI
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # -------------------------------
    # 7) ATR(14) on High, Low, Close
    # -------------------------------
    # Compute True Range components as separate Series
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    # TrueRange = row‐wise max of tr1, tr2, tr3
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR_14 = rolling mean of TrueRange
    df["ATR_14"] = true_range.rolling(window=14, min_periods=14).mean()

    # -------------------------------
    # 8) Volume lag(1)
    # -------------------------------
    df["Volume_lag1"] = df["Volume"].shift(1)

    # -------------------------------
    # 9) Now drop any rows where these new columns are NaN
    # -------------------------------
    #   We know exactly which columns we just created; all must exist now:
    to_dropna = ["Return", "EMA_10", "SMA_20", "RSI_14", "ATR_14", "Volume_lag1"]
    #   Double‐check they are in df.columns (they should be, since we just assigned them)
    missing_new = set(to_dropna) - set(df.columns)
    if missing_new:
        raise KeyError(f"Expected to find newly created columns {to_dropna}, but missing {missing_new}")

    df = df.dropna(subset=to_dropna).reset_index(drop=True)
    return df
    # final df columns: [
    #   'date','Open','High','Low','Close','Volume',
    #   'Return','EMA_10','SMA_20','RSI_14','ATR_14','Volume_lag1'
    # ]


def merge_sentiment_and_price(
    df_daily_sent: pd.DataFrame, df_tech: pd.DataFrame
) -> pd.DataFrame:
    """
    1) Merge df_daily_sent (['date','label','pos_avg','neu_avg','neg_avg'])
       with df_tech (['date','Open','High','Low','Close','Volume','Return','EMA_10','SMA_20','RSI_14','ATR_14','Volume_lag1']).
    2) The final DataFrame has columns:
       ['date','label','pos_avg','neu_avg','neg_avg',
        'Open','High','Low','Close','Volume','Return',
        'EMA_10','SMA_20','RSI_14','ATR_14','Volume_lag1']
    3) Drop any rows where merging failed (but in theory every date in news should exist in price).
    Returns that merged DataFrame, ready for modeling.
    """
    df_merged = pd.merge(
        df_daily_sent,
        df_tech,
        how="inner",
        on="date",
        validate="one_to_one"
    )
    return df_merged