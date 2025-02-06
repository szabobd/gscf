import yfinance as yf
import pandas as pd
from typing import List
import datetime
from pathlib import Path


class DataLoader:
    """Handles downloading and saving stock data."""
    def __init__(self, tickers: List[str], start_date: datetime.date, end_date: datetime.date, data_dir: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def fetch_api_and_save_to_csv(self) -> None:
        """Fetches stock data and saves as CSV."""
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            file_path = self.data_dir / f"{ticker}.csv"
            df.to_csv(file_path)
            print(f"Saved {ticker} data to {file_path}")
    
    @staticmethod
    def load_reindexed_csv(filename: str, data_dir: str) -> pd.DataFrame:
        return pd.read_csv(
            Path(data_dir) / filename,
            index_col=["Date", "Ticker"], 
            parse_dates=["Date"], 
            date_format="%Y-%m-%d"
        )