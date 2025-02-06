import pandas as pd
from typing import List, Dict, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.axes._axes import Axes
import seaborn as sns
from my_utils.data_loader import DataLoader



class Visualizer:
    def __init__(self, color_palette: Dict[str, tuple]):
        self.color_palette = color_palette

    def visualize(path: str, tickers: List[str], filename_with_analysis: str) -> None:
        df = DataLoader.load_reindexed_csv(filename_with_analysis, path)
        visualizer = Visualizer(color_palette=dict(zip(tickers, plt.get_cmap('tab10', len(tickers)).colors)))
        visualizer.plot_closing_prices(df)
        visualizer.plot_daily_returns(df)
        visualizer.plot_moving_averages(df)

    @staticmethod
    def style_plot(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            plt.style.use("seaborn-v0_8")
            fig, ax = plt.subplots(figsize=(14, 7))
            func(*args, ax=ax, **kwargs)
            Visualizer._finalize_plot(ax)
            plt.show()
        return wrapper

    @staticmethod
    def _configure_xaxis(ax: Axes) -> None:
        """Configures the x-axis formatting for date displays."""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

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

    @style_plot
    def plot_closing_prices(self, df: pd.DataFrame, ax: Optional[Axes] = None) -> None:
        sns.lineplot(
            data=df,
            x=df.index.get_level_values("Date"),
            y="Close",
            hue=df.index.get_level_values("Ticker"),
            ax=ax,
            palette=self.color_palette
        )
        ax.set_title("Stock Closing Prices", fontsize=16, weight='bold')

    def plot_daily_returns(self, df: pd.DataFrame, ax: Optional[Axes] = None) -> None:
        tickers = df.index.get_level_values("Ticker").unique()
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
        fig.suptitle("Daily Returns for Each Stock", fontsize=16, weight='bold')

        for i, ticker in enumerate(tickers):
            ax = axes[i // 3, i % 3]
            sns.lineplot(
                data=df.xs(ticker, level="Ticker")["Daily Return"],
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

        for j in range(i + 1, 6):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @style_plot
    def plot_moving_averages(self, df: pd.DataFrame, ax: Optional[Axes] = None) -> None:
        tickers = df.index.get_level_values("Ticker").unique()
        for ticker in tickers:
            stock_df = df.xs(ticker, level="Ticker")
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
        # x-axis configuration is handled by _finalize_plot via the style_plot decorator

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

    def _plot_crossovers(self, ax: Axes, stock_df: pd.DataFrame, ticker: str, color: str) -> None:
        """Plots positive and negative crossovers for a given stock."""
        positive = stock_df[stock_df["Crossover"] == 1]
        negative = stock_df[stock_df["Crossover"] == -1]
        self._plot_scatter(ax, positive, ticker, color, '^', 'Positive Crossover')
        self._plot_scatter(ax, negative, ticker, color, 'v', 'Negative Crossover')