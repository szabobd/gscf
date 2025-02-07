import pytest
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
import pandas as pd
import datetime
from my_utils.data_loader import DataLoader

@pytest.fixture
def setup_data_loader():
    tickers = ['AAPL', 'GOOG']
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    data_dir = 'test_data'
    return DataLoader(tickers, start_date, end_date, data_dir)

# fetch_api_and_save_to_csv function tests

@patch('my_utils.data_loader.yf.download')
def test_fetch_api_and_save_to_csv_handles_empty_dataframe(mock_download, setup_data_loader, capfd):
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_download.return_value = mock_df
    mock_df.empty = True
    
    with pytest.raises(SystemExit):
        setup_data_loader.fetch_api_and_save_to_csv()
    
    captured = capfd.readouterr()
    mock_df.to_csv.assert_not_called()
    assert "No data found for AAPL, please only provide valid tickers" in captured.out

@patch('my_utils.data_loader.yf.download')
def test_fetch_api_and_save_to_csv_handles_exception(mock_download, setup_data_loader, capfd):
    mock_download.side_effect = Exception("Failed to fetch data")
    
    setup_data_loader.fetch_api_and_save_to_csv()
    
    captured = capfd.readouterr()
    assert "Failed to fetch or save data for AAPL: Failed to fetch data" in captured.out


@patch('my_utils.data_loader.yf.download')
def test_fetch_api_and_save_to_csv(mock_download, setup_data_loader):
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_download.return_value = mock_df
    mock_df.empty = False
    
    setup_data_loader.fetch_api_and_save_to_csv()
    
    for ticker in setup_data_loader.tickers:
        mock_download.assert_any_call(ticker, start=setup_data_loader.start_date, end=setup_data_loader.end_date)
        file_path = Path(setup_data_loader.data_dir) / f"{ticker}.csv"
        mock_df.to_csv.assert_any_call(file_path)
        print(f"Tested saving {ticker} data to {file_path}")

# load_reindexed_csv function tests

@patch('my_utils.data_loader.pd.read_csv')
def test_load_reindexed_csv(mock_read_csv):
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_read_csv.return_value = mock_df

    filename = 'test.csv'
    data_dir = 'test_data'
    result = DataLoader.read_reindexed_csv(filename, data_dir)

    mock_read_csv.assert_called_once_with(
        Path(data_dir) / filename,
        index_col=["Date", "Ticker"],
        parse_dates=["Date"],
        date_format="%Y-%m-%d"
    )

    assert result == mock_df

def test_load_reindexed_csv_file_not_found():
    filename = 'non_existent_file.csv'
    data_dir = 'test_data'

    with pytest.raises(FileNotFoundError):
        DataLoader.read_reindexed_csv(filename, data_dir)

@patch('my_utils.data_loader.pd.read_csv')
def test_load_reindexed_csv_invalid_format(mock_read_csv):
    filename = 'invalid_format.csv'
    data_dir = 'test_data'
    mock_read_csv.side_effect = pd.errors.ParserError("Invalid format")

    with pytest.raises(pd.errors.ParserError):
        DataLoader.read_reindexed_csv(filename, data_dir)

@patch('my_utils.data_loader.pd.read_csv')
def test_load_reindexed_csv_general_exception(mock_read_csv):
    filename = 'test.csv'
    data_dir = 'test_data'
    mock_read_csv.side_effect = Exception("General error")

    with pytest.raises(Exception, match="General error"):
        DataLoader.read_reindexed_csv(filename, data_dir)


if __name__ == '__main__':
    pytest.main()
