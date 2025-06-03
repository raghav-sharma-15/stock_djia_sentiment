# src/data_loader.py

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def load_and_melt_djia_news(csv_path: str) -> pd.DataFrame:
    """
    1) Loads Combined_News_DJIA.csv (columns: Date, Label, Top1…Top25).
    2) Melts Top1…Top25 into individual rows: each headline has its 'Date' & 'Label'.
    3) Strips leading b"…", b'…' wrappers from the headline text.
    Returns a DataFrame with columns: ['date', 'label', 'headline'].
    """
    # 1) Load the raw CSV
    df_raw = pd.read_csv(csv_path, parse_dates=["Date"])
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)

    # 2) Melt Top1…Top25 into a single 'headline' column
    top_cols = [f"Top{i}" for i in range(1, 26)]
    df_melted = df_raw.melt(
        id_vars=["Date", "Label"],
        value_vars=top_cols,
        var_name="Rank",
        value_name="raw_headline"
    )

    # 3) Clean up each raw_headline: remove leading b" or b' and trailing quote
    def strip_b_wrapper(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        # It looks like: b"Some headline here"  or  b'Some headline here'
        if s.startswith("b\"") and s.endswith("\""):
            return s[2:-1]
        if s.startswith("b'") and s.endswith("'"):
            return s[2:-1]
        return s

    df_melted["headline"] = df_melted["raw_headline"].apply(strip_b_wrapper)
    df_melted = df_melted.rename(columns={"Date": "date", "Label": "label"})
    df_melted = df_melted[["date", "label", "headline"]].copy()
    df_melted = df_melted.dropna(subset=["headline"]).reset_index(drop=True)
    return df_melted  # columns: date, label, headline


def fetch_djia_price(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads DJIA index (^DJI) daily OHLCV from start_date to end_date (inclusive).
    Ensures the returned DataFrame has columns named exactly ['date','Open','High','Low','Close','Volume'].
    """
    df_price = yf.download("^DJI", start=start_date, end=end_date, interval="1d", progress=False)

    # If yf.download returns a MultiIndex (e.g. columns = MultiIndex([('Open','^DJI'), ...])),
    # flatten it to single level by taking only the first level (field names).
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)

    df_price = df_price.reset_index()
    df_price = df_price.rename(columns={"Date": "date"})
    # Keep only the required columns and ensure they exist
    expected = ["date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df_price.columns]
    if missing:
        raise KeyError(f"fetch_djia_price expected columns {expected}, but missing {missing}")
    return df_price[expected].copy()


def build_price_dataframe(dates_series: pd.Series) -> pd.DataFrame:
    """
    Given a Series of dates (from the news dataset), determine min_date and max_date,
    fetch DJIA prices from min_date to (max_date + 1 day), returns that price DataFrame.
    """
    min_date = dates_series.min().strftime("%Y-%m-%d")
    # Add one extra day so that technicals can use next-day returns if needed
    max_date = (dates_series.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df_price = fetch_djia_price(min_date, max_date)
    # Ensure 'date' is datetime
    df_price["date"] = pd.to_datetime(df_price["date"])
    return df_price