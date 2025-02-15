from pathlib import Path
from typing import Type

import pandas as pd

from my_utils.data_loader import DataLoader


class Analyzer:
    """Performs time-series analysis."""
       
    @classmethod
    def analyze_data(cls: Type['Analyzer'], data_dir: Path, input_filename: str, output_filename: str) -> pd.DataFrame:
        """Performs time-series analysis on stock data."""       
        df = DataLoader.read_reindexed_csv(input_filename, data_dir)
        df = cls._detect_crossovers(df)

        output_path = data_dir / output_filename
        df.to_csv(output_path)

        print("Analysis completed. Data saved to:", output_path)
        return df
    
    @staticmethod
    def _detect_crossovers(df: pd.DataFrame) -> pd.DataFrame:
        def compute_crossover(group: pd.DataFrame) -> pd.Series:
            pos_crossover = (group["30D Rolling Avg"] > group["14D EMA"]) & \
                            (group["30D Rolling Avg"].shift(1) <= group["14D EMA"].shift(1))
        
            neg_crossover = (group["30D Rolling Avg"] < group["14D EMA"]) & \
                            (group["30D Rolling Avg"].shift(1) >= group["14D EMA"].shift(1))
            
            return pos_crossover.astype(int) - neg_crossover.astype(int)
        
        df["Crossover"] = df.groupby("Ticker").apply(compute_crossover).reset_index(level=0, drop=True)
        
        # Shift the signal by one period so that the signal occurs on the day following the crossover.
        df["Crossover"] = df.groupby("Ticker")["Crossover"].shift(1, fill_value=0)

        return df
       