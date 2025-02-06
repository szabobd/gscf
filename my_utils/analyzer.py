import pandas as pd
from my_utils.data_loader import DataLoader
from typing import Type
from pathlib import Path


class Analyzer:
    """Performs time-series analysis."""
       
    @classmethod
    def analyze_data(cls: Type['Analyzer'], data_dir: Path, input_filename: str, output_filename: str) -> pd.DataFrame:
        """Performs time-series analysis on stock data."""       

        df = DataLoader.load_reindexed_csv(input_filename, data_dir)
        df = cls.detect_crossovers(df)

        output_path = data_dir / output_filename
        df.to_csv(output_path)

        print("Analysis completed. Data saved to:", output_path)
        return df
    
    @staticmethod
    def detect_crossovers(df: pd.DataFrame) -> pd.DataFrame:
        df["Crossover"] = df.groupby('Ticker').apply(
            lambda group: ((group["30D Rolling Avg"] > group["14D EMA"]) & 
                           (group["30D Rolling Avg"].shift(1) <= group["14D EMA"].shift(1))).astype(int) - 
                          ((group["30D Rolling Avg"] < group["14D EMA"]) & 
                           (group["30D Rolling Avg"].shift(1) >= group["14D EMA"].shift(1))).astype(int)
        ).reset_index(level=0, drop=True)
        df["Crossover"] = df.groupby('Ticker')['Crossover'].shift(1, fill_value=0)
        print(df['Crossover'].value_counts())
        return df