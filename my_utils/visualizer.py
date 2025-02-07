from typing import Dict, Optional, Callable

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.axes._axes import Axes
import seaborn as sns

from my_utils.data_loader import DataLoader


class Visualizer:
    def __init__(self, color_palette: Dict[str, tuple]):
        self.color_palette = color_palette

    def visualize(self, path: str, filename_with_analysis: str) -> None:
        """Orchestrates visualization of stock data."""
        self.df = DataLoader.read_reindexed_csv(filename_with_analysis, path)
        self._plot_closing_prices()
        self._plot_daily_returns()
        self._plot_moving_averages()

    @staticmethod
    def _style_and_draw_plot(func: Callable) -> Callable:
        """Decorator to adjust style and draw plot."""
        def wrapper(*args, **kwargs):
            plt.style.use("seaborn-v0_8")
            fig, ax = plt.subplots(figsize=(14, 7))
            func(*args, ax=ax, **kwargs)
            Visualizer._finalize_plot(ax)
            plt.show()

        return wrapper

    @staticmethod
    def _finalize_plot(ax: Axes) -> None:
        """Finalizes the plot by adding labels, legend, and applying tight layout."""
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        Visualizer._configure_xaxis(ax)
        handles, labels = ax.get_legend_handles_labels()

        ax.legend(dict(zip(labels, handles)).values(),
                  dict(zip(labels, handles)).keys(),
                  loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
        
        plt.tight_layout()

    @staticmethod
    def _configure_xaxis(ax: Axes) -> None:
        """Configures the x-axis formatting for date displays."""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    @_style_and_draw_plot
    def _plot_closing_prices(self, ax: Optional[Axes] = None) -> None:
        """ Sets up plot to visualize closing prices."""
        sns.lineplot(
            data=self.df,
            x=self.df.index.get_level_values("Date"),
            y="Close",
            hue=self.df.index.get_level_values("Ticker"),
            ax=ax,
            palette=self.color_palette
        )
        ax.set_title("Stock Closing Prices", fontsize=16, weight='bold')

    def _plot_daily_returns(self) -> None:
        """ Sets up plot to visualize daily returns."""
        tickers = self.df.index.get_level_values("Ticker").unique()
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
        fig.suptitle("Daily Returns for Each Stock", fontsize=16, weight='bold')

        for i, ticker in enumerate(tickers):
            ax = axes[i // 3, i % 3]
            self._draw_grid_plots(ax, ticker)

        # Removing unused subplots
        for j in range(i + 1, 6):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @_style_and_draw_plot
    def _plot_moving_averages(self, ax: Optional[Axes] = None) -> None:
        """ Sets up plot to visualize moving averages and crossovers."""
        tickers = self.df.index.get_level_values("Ticker").unique()

        for ticker in tickers:
            stock_df = self.df.xs(ticker, level="Ticker")
            color = self.color_palette[ticker]
            ax.plot(
                stock_df.index,
                stock_df["30D Rolling Avg"],
                label=f"{ticker} 30-Day SMA",
                linestyle="--",
                linewidth=2,
                color=color
            )
            ax.plot(
                stock_df.index,
                stock_df["14D EMA"],
                label=f"{ticker} 14D EMA",
                linestyle=":",
                linewidth=2,
                color=color
            )
            self._plot_crossovers(ax, stock_df, ticker, color)
        ax.set_title("30 Day Rolling Avg & 14 Day EMA with Crossovers", fontsize=16, weight='bold')

    def _draw_grid_plots(self, ax: Axes, ticker: str) -> None:
        """Draws the grid of plots for daily returns."""
        sns.lineplot(
            data=self.df.xs(ticker, level="Ticker")["Daily Return"],
            ax=ax,
            color=self.color_palette[ticker]
        )
        ax.set_title(f"{ticker} Daily Returns", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Daily Return (%)", fontsize=12)
        self._configure_xaxis(ax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        for label in ax.get_xticklabels():
            label.set_visible(True)
            label.set_rotation(45)

    def _plot_crossovers(self, ax: Axes, stock_df: pd.DataFrame, ticker: str, color: str) -> None:
        """Plots positive and negative crossovers for a given stock."""
        positive = stock_df[stock_df["Crossover"] == 1]
        negative = stock_df[stock_df["Crossover"] == -1]
        self._plot_scatter(ax, positive, ticker, color, '^', 'Positive Crossover')
        self._plot_scatter(ax, negative, ticker, color, 'v', 'Negative Crossover')

    def _plot_scatter(self, ax: Axes, data: pd.DataFrame, ticker: str, color: str, marker: str, label_suffix: str) -> None:
        """Helper to plot scatter data for crossovers."""
        if not data.empty:
            ax.scatter(
                data.index,
                data["Close"],
                color=color,
                marker=marker,
                label=f'{ticker} {label_suffix}',
                s=20,
                edgecolor='black',
                zorder=5
            )
    