import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from my_utils.analyzer import Analyzer
from my_utils.data_loader import DataLoader

@patch('my_utils.analyzer.DataLoader.read_reindexed_csv')
@patch.object(Analyzer, 'detect_crossovers')
def test_analyze_data(mock_detect_crossovers, mock_load_reindexed_csv):
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_load_reindexed_csv.return_value = mock_df
    mock_detect_crossovers.return_value = mock_df
    
    data_dir = Path('test_data')
    input_filename = 'input.csv'
    output_filename = 'output.csv'

    result = Analyzer.analyze_data(data_dir, input_filename, output_filename)

    mock_load_reindexed_csv.assert_called_once_with(input_filename, data_dir)
    mock_detect_crossovers.assert_called_once_with(mock_df)

    mock_df.to_csv.assert_called_once_with(data_dir / output_filename)

    assert result == mock_df

def test_detect_crossovers_with_crossover():
    test_data = create_test_data_with_crossover()

    result = Analyzer.detect_crossovers(test_data)

    expected_crossover = pd.Series([0, 0, 0, -1, 1, 0, 0, 0, -1, 1], name="Crossover", index=test_data.index)
    pd.testing.assert_series_equal(result["Crossover"], expected_crossover)

    
def test_detect_crossovers_without_crossover():
    test_data = create_test_data_without_crossover()

    result = Analyzer.detect_crossovers(test_data)

    expected_crossover = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], name="Crossover", index=test_data.index)
    pd.testing.assert_series_equal(result["Crossover"], expected_crossover)

def create_test_data_with_crossover() -> pd.DataFrame:
    data = {
        "Ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "NVDA", "NVDA", "NVDA", "NVDA", "NVDA"],
        "30D Rolling Avg": [100, 105, 10, 115, 120, 100, 105, 10, 115, 120],
        "14D EMA": [90, 95, 100, 105, 110, 90, 95, 100, 105, 110],
        "Date": pd.date_range(start="2020-01-01", periods=10)
    }

    df = pd.DataFrame(data)
    df.set_index(['Date', 'Ticker'], inplace=True)

    return df

def create_test_data_without_crossover() -> pd.DataFrame:
    data = {
        "Ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "NVDA", "NVDA", "NVDA", "NVDA", "NVDA"],
        "30D Rolling Avg": [100, 105, 105, 115, 120, 100, 105, 115, 115, 120],
        "14D EMA":         [90, 95, 100, 105, 110, 90, 95, 100, 105, 110],
        "Date": pd.date_range(start="2020-01-01", periods=10)
    }

    df = pd.DataFrame(data)
    df.set_index(['Date', 'Ticker'], inplace=True)

    return df
