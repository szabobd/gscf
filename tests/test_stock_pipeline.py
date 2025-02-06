import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os
from typer.testing import CliRunner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import DataLoader, Transformer, Analyzer, Visualizer, main


def test_set_up(self):
    self.tickers = ["AMD", "NVDA", "INTC", "MSFT", "AMZN"]
    self.start_date = "2020-01-01"
    self.end_date = "2023-12-31"
    self.data_dir = Path("test_data")
    self.data_dir.mkdir(exist_ok=True)

@patch('yfinance.download')
def test_data_loader(self, mock_download):
    mock_df = pd.DataFrame({
        "Date": pd.date_range(start=self.start_date, periods=5),
        "Close": [100, 101, 102, 103, 104]
    })
    mock_download.return_value = mock_df

    loader = DataLoader(self.tickers, self.start_date, self.end_date, data_dir=self.data_dir)
    loader.fetch_and_save()

    for ticker in self.tickers:
        file_path = self.data_dir / f"{ticker}.csv"
        self.assertTrue(file_path.exists())

def test_transformer(self):
    for ticker in self.tickers:
        mock_df = pd.DataFrame({
            "Date": pd.date_range(start=self.start_date, periods=5),
            "Close": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [95, 96, 97, 98, 99],
            "Open": [98, 99, 100, 101, 102],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        })
        mock_df.to_csv(self.data_dir / f"{ticker}.csv", index=False)

    transformer = Transformer(data_dir=self.data_dir)
    df = transformer.load_and_transform(self.tickers)

    self.assertFalse(df.empty)
    self.assertIn("Daily Return", df.columns)
    self.assertIn("30D Rolling Avg", df.columns)

def test_analyzer(self):
    mock_data = {
        "Date": pd.date_range(start=self.start_date, periods=5),
        "Ticker": ["AMD"] * 5,
        "Close": [100, 101, 102, 103, 104],
        "30D Rolling Avg": [100, 101, 102, 103, 104],
        "14D EMA": [99, 100, 101, 102, 103]
    }
    df = pd.DataFrame(mock_data).set_index(["Ticker", "Date"])
    analyzer = Analyzer()
    analyzed_df = analyzer.detect_crossovers(df)

    self.assertIn("Crossover", analyzed_df.columns)
    self.assertTrue((analyzed_df["Crossover"] != 0).any())

@patch('matplotlib.pyplot.show')
def test_visualizer(self, mock_show):
    mock_data = {
        "Date": pd.date_range(start=self.start_date, periods=5),
        "Ticker": ["AMD"] * 5,
        "Close": [100, 101, 102, 103, 104],
        "30D Rolling Avg": [100, 101, 102, 103, 104],
        "14D EMA": [99, 100, 101, 102, 103],
        "Crossover": [0, 1, 0, -1, 0]
    }
    df = pd.DataFrame(mock_data).set_index(["Ticker", "Date"])

    Visualizer.plot_closing_prices(df)
    Visualizer.plot_daily_returns(df)
    Visualizer.plot_moving_averages(df)

    self.assertTrue(mock_show.called)

def test_tear_down(self):
    for file in self.data_dir.glob("*.csv"):
        file.unlink()
    self.data_dir.rmdir()

@patch('yfinance.download')
def test_invalid_ticker(self, mock_download):
    mock_download.return_value = pd.DataFrame()
    loader = DataLoader(['INVALID'], '2022-01-01', '2022-12-31', data_dir='./test_data')
    loader.fetch_and_save()
    file_path = Path('./test_data/INVALID.csv')
    self.assertTrue(file_path.exists())
    df = pd.read_csv(file_path)
    self.assertTrue(df.empty, "The DataFrame should be empty for invalid tickers.")

def test_transform_with_missing_values(self):
    df = pd.DataFrame({
        'Date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
        'Close': [100, None, 102, None, 104],
        'Volume': [1000, 1100, None, 1300, 1400],
        'Ticker': ['TEST'] * 5
    })
    df.to_csv('./test_data/TEST.csv', index=False)
    transformer = Transformer(data_dir='./test_data')
    transformed_df = transformer.load_and_transform(['TEST'])

    self.assertFalse(transformed_df['Close'].isnull().any(), "Missing Close values should be filled.")
    self.assertFalse(transformed_df['Volume'].isnull().any(), "Missing Volume values should be handled.")

def test_analyzer_no_crossovers(self):
    df = pd.DataFrame({
        'Date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
        'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        '30D Rolling Avg': [100] * 10,
        '14D EMA': [110] * 10,
        'Ticker': ['TEST'] * 10
    })
    df.set_index(['Ticker', 'Date'], inplace=True)
    analyzer = Analyzer()
    analyzed_df = analyzer.detect_crossovers(df)
    self.assertEqual(analyzed_df['Crossover'].sum(), 0, "There should be no crossovers detected.")

@patch('matplotlib.pyplot.show')
def test_visualizer_with_empty_data(self, mock_show):
    empty_df = pd.DataFrame(columns=['Date', 'Close', 'Ticker']).set_index(['Ticker', 'Date'])
    with self.assertRaises(ValueError):
        Visualizer.plot_closing_prices(empty_df)

def test_cli_full_pipeline(self):
    runner = CliRunner()
    result = runner.invoke(main, ['--full'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn("Analysis completed and saved.", result.output)

#----------------------------------------------------------------------------------------------------------------------------

class TestETL(unittest.TestCase):
    def test_data_loader(self):
        loader = DataLoader(["AAPL"], "2023-01-01", "2023-12-31", "./data")
        self.assertIsInstance(loader.tickers, list)

    def test_transformer(self):
        transformer = Transformer("./data")
        sample_data = pd.DataFrame({"Close": [100, 102, 104], "Volume": [1000, None, 1200]}, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
        sample_data["Volume"].bfill(inplace=True)
        self.assertFalse(sample_data["Volume"].isna().any())

    def test_crossovers(self):
        sample_data = pd.DataFrame({"30D Rolling Avg": [100, 102, 98], "14D EMA": [101, 101, 99], "Ticker": ["AAPL"] * 3})
        sample_data.set_index(["Ticker", pd.date_range("2023-01-01", periods=3)], inplace=True)
        df = Analyzer.detect_crossovers(sample_data)
        self.assertListEqual(df["Crossover"].tolist(), [0, -1, 1])