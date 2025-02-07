from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from my_utils.visualizer import Visualizer

@patch("my_utils.data_loader.DataLoader.read_reindexed_csv")
def test_visualize_calls_correct_methods(mock_load_csv):
    """Ensure visualize method correctly calls internal plotting functions."""
    mock_load_csv.return_value = sample_data
    viz = Visualizer(color_palette)

    with patch("matplotlib.pyplot.show"), \
         patch.object(viz, "_plot_closing_prices") as mock_closing, \
         patch.object(viz, "_plot_daily_returns") as mock_daily, \
         patch.object(viz, "_plot_moving_averages") as mock_moving:

        viz.visualize("mock_path", "mock_filename.csv")

        mock_closing.assert_called_once()
        mock_daily.assert_called_once()
        mock_moving.assert_called_once()

sample_data = pd.DataFrame({
    "Date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
    "Ticker": ["AAPL"] * 10,
    "Close": [150 + i for i in range(10)],
    "30D Rolling Avg": [150 + i for i in range(10)],
    "14D EMA": [150 + i for i in range(10)],
    "Daily Return": [0.01] * 10,
    "Crossover": [0] * 8 + [1, -1]
}).set_index(["Date", "Ticker"])

color_palette = {"AAPL": "blue"}

def test_plot_closing_prices():
    """Test closing prices plot without displaying the figure."""
    viz = Visualizer(color_palette)
    viz.df = sample_data

    with patch("matplotlib.pyplot.show"), \
         patch("seaborn.lineplot") as mock_lineplot:
        viz._plot_closing_prices()
        mock_lineplot.assert_called_once()

def test_plot_moving_averages():
    """Test moving averages plot without displaying the figure."""
    viz = Visualizer(color_palette)
    viz.df = sample_data
    fig, ax = plt.subplots()

    with patch.object(viz, "_plot_crossovers") as mock_crossovers, \
         patch("matplotlib.pyplot.show"):

        viz._plot_moving_averages()

        tickers = viz.df.index.get_level_values("Ticker").unique()

        assert mock_crossovers.call_count == len(tickers)

def test_plot_daily_returns():
    """Test daily returns plot without displaying the figure."""
    viz = Visualizer(color_palette)
    viz.df = sample_data
    mock_axes = np.array([MagicMock(spec=plt.Axes) for _ in range(6)]).reshape(2, 3)

    with patch("matplotlib.pyplot.show"), \
         patch.object(viz, "_draw_grid_plots") as mock_draw_grid, \
            patch("matplotlib.pyplot.subplots") as mock_subplots, \
                patch("matplotlib.pyplot.tight_layout") as mock_tight_layout, \
                    patch("matplotlib.figure.Figure.delaxes") as mock_delaxes:
        mock_subplots.return_value = (plt.figure(), mock_axes)
        viz._plot_daily_returns()
        mock_draw_grid.assert_called_once()
        mock_subplots.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_delaxes.assert_called()
        

def test_plot_crossovers():
    """Test crossover points are plotted."""
    viz = Visualizer(color_palette)
    fig, ax = plt.subplots()
    testDf = sample_data.xs("AAPL", level="Ticker")

    with patch.object(viz, "_plot_scatter") as mock_plot_scatter:
        viz._plot_crossovers(ax, testDf, "AAPL", "blue")

        assert mock_plot_scatter.call_count == 2

        calls = mock_plot_scatter.call_args_list

        df1 = calls[0][0][1]  # First call, second argument (DataFrame)
        df2 = calls[1][0][1]  # Second call, second argument (DataFrame)

        expected_positive = testDf[testDf["Crossover"] == 1]
        expected_negative = testDf[testDf["Crossover"] == -1]

        assert_frame_equal(df1, expected_positive)
        assert_frame_equal(df2, expected_negative)

def test_draw_grid_plots():
    """Test grid plot for daily returns."""
    viz = Visualizer(color_palette)
    mock_df = MagicMock(spec=pd.DataFrame)
    viz.df = mock_df
    fig, ax = plt.subplots()

    with patch("seaborn.lineplot") as mock_lineplot:
        viz._draw_grid_plots(ax, "AAPL")
        mock_lineplot.assert_called_once()
