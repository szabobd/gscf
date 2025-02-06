import pandas as pd
from typing import List
from pathlib import Path


class Transformer:
    """Handles data cleaning and transformations."""
    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

    def load_and_transform(self, tickers: List[str], filename_merged: str) -> pd.DataFrame:
        """Loads, cleans, and transforms data."""
        all_data = [self.__get_cleaned_ticker_csv(ticker) for ticker in tickers]
        self.df = pd.concat(all_data)
        self.__calculate_metrics()
        output_path = self.data_dir / filename_merged
        self.df.to_csv(output_path)
        print(f"Transformed data saved to {output_path}")
        return self.df

    def __get_cleaned_ticker_csv(self, ticker: str) -> pd.DataFrame:
        file_path = self.data_dir / f"{ticker}.csv"
        df = pd.read_csv(file_path)
        return self.__clean_ticker_data(df, ticker)
        
    
    def __clean_ticker_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = df.iloc[2:].reset_index(drop=True)
        df['Ticker'] = ticker
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index(['Ticker', 'Date'], inplace=True)
        df.sort_index(inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        return df
    

    def __calculate_metrics(self) -> None:
        self.df["Daily Return"] = self.df.groupby('Ticker')['Close'].pct_change()
        self.df["30D Rolling Avg"] = self.df.groupby('Ticker')['Close'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)
        self.df["14D EMA"] = self.df.groupby('Ticker')['Close'].ewm(span=14, adjust=False).mean().reset_index(level=0, drop=True)
        self.df["Volatility"] = self.df.groupby('Ticker')['Daily Return'].rolling(window=10, min_periods=1).std().reset_index(level=0, drop=True)
        self.df['Close'] = self.df.groupby('Ticker')['Close'].ffill()
        self.df['Volume'] = self.df.groupby('Ticker')['Volume'].bfill()
