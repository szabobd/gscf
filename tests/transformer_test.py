import pandas as pd
from unittest.mock import patch

from my_utils.transformer import Transformer


@patch.object(Transformer, '_get_cleaned_ticker_csv')
@patch.object(Transformer, '_calculate_metrics')
@patch('my_utils.transformer.pd.DataFrame.to_csv')
def test_read_and_transform(mock_to_csv, mock_calculate_metrics, mock_get_cleaned_ticker_csv):
    # Sample data for testing
    data1 = {
        'Date': pd.date_range(start='2022-01-01', periods=3, freq='D'),
        'Close': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [95, 96, 97],
        'Open': [98, 99, 100],
        'Volume': [1000, 1100, 1200],
        'Ticker': ['A', 'A', 'A']
    }
    data2 = {
        'Date': pd.date_range(start='2022-01-01', periods=3, freq='D'),
        'Close': [200, 201, 202],
        'High': [205, 206, 207],
        'Low': [195, 196, 197],
        'Open': [198, 199, 200],
        'Volume': [2000, 2100, 2200],
        'Ticker': ['B', 'B', 'B']
    }
    df1 = pd.DataFrame(data1).set_index(['Ticker', 'Date'])
    df2 = pd.DataFrame(data2).set_index(['Ticker', 'Date'])

    mock_get_cleaned_ticker_csv.side_effect = [df1, df2]

    transformer = Transformer(data_dir='.')
    result_df = transformer.read_and_transform(tickers=['A', 'B'], filename_merged='merged.csv')

    mock_get_cleaned_ticker_csv.assert_any_call('A')
    mock_get_cleaned_ticker_csv.assert_any_call('B')

    mock_calculate_metrics.assert_called_once()

    mock_to_csv.assert_called_once_with(transformer.data_dir / 'merged.csv')

    expected_df = pd.concat([df1, df2])
    pd.testing.assert_frame_equal(result_df, expected_df)


@patch('my_utils.transformer.pd.read_csv')
@patch.object(Transformer, '_clean_ticker_data')
def test_get_cleaned_ticker_csv(mock_clean_ticker_data, mock_read_csv):
    df = get_test_ticker_data()
    indexed_df = get_indexed_test_ticker_data()

    mock_read_csv.return_value = df
    mock_clean_ticker_data.return_value = indexed_df

    transformer = Transformer(data_dir='.')
    cleaned_df = transformer._get_cleaned_ticker_csv("A")
    
    mock_read_csv.assert_called_once_with(transformer.data_dir / "A.csv")
    pd.testing.assert_frame_equal(cleaned_df, indexed_df)


def test_clean_ticker_data():
    df = get_test_ticker_data()
    ticker = 'A'
    
    cleaned_df = Transformer._clean_ticker_data(df, ticker)
    
    assert cleaned_df.index.names == ['Ticker', 'Date']
    assert cleaned_df['Close'].dtype == 'float64'
    assert cleaned_df['Volume'].dtype == 'int64'
    assert cleaned_df.index.get_level_values('Ticker').unique() == [ticker]
    assert cleaned_df.index.get_level_values('Date').dtype == 'datetime64[ns]'
    assert cleaned_df.isna().sum().sum() == 0


def test_calculate_metrics():
    data = {
        'Ticker': ['A', 'A', 'A', 'B', 'B', 'B'],
        'Date': pd.date_range(start='2022-01-01', periods=6, freq='D'),
        'Close': [100, 101, 102, 200, 202, 204],
        'Volume': [1000, 1100, 1200, 2000, 2100, 2200]
    }
    df = pd.DataFrame(data)
    df.set_index(['Ticker', 'Date'], inplace=True)
    
    transformer = Transformer(data_dir='.')
    transformer.df = df.copy()
    transformer._calculate_metrics()
    
    assert 'Daily Return' in transformer.df.columns
    assert '30D Rolling Avg' in transformer.df.columns
    assert '14D EMA' in transformer.df.columns
    assert 'Volatility' in transformer.df.columns
    assert 'Close' in transformer.df.columns
    assert 'Volume' in transformer.df.columns
    
    assert transformer.df['30D Rolling Avg'].notna().all()
    assert transformer.df['14D EMA'].notna().all()
    assert transformer.df['Close'].notna().all()
    assert transformer.df['Volume'].notna().all()
    
    assert transformer.df['Close'].isna().sum() == 0
    assert transformer.df['Volume'].isna().sum() == 0


def get_test_ticker_data() -> pd.DataFrame:
    data = {
        'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04'],
        'Close': ['100.1', '101.1', '102.1', '103.1'],
        'High': ['105', '106', '107', '108'],
        'Low': ['95', '96', '97', '98'],
        'Open': ['98', '99', '100', '101'],
        'Volume': ['1000', '1100', '1200', '1300']
    }
    return pd.DataFrame(data)


def get_indexed_test_ticker_data() -> pd.DataFrame:
    df = get_test_ticker_data()
    df.set_index('Date', inplace=True)
    return df
