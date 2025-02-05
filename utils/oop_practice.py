import yfinance as yf
import pandas as pd
import typer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
import unittest


class DataLoader:
    """Handles downloading and saving stock data."""
    def __init__(self, tickers, start_date, end_date, data_dir):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def fetch_and_save(self):
        """Fetches stock data and saves as CSV."""
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date)

            file_path = self.data_dir / f"{ticker}.csv"
            df.to_csv(file_path)
            print(f"Saved {ticker} data to {file_path}")

class Transformer:
    """Handles data cleaning and transformations."""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load_and_transform(self, tickers):
        """Loads, cleans, and transforms data."""
        all_data = []
        
        for ticker in tickers:
            file_path = self.data_dir / f"{ticker}.csv"
            df = pd.read_csv(file_path)
            
            # Extract the ticker value and clean the data
            ticker_value = df.iloc[0, 1]  # Ticker value in first row (second column)
            df = df.iloc[2:].reset_index(drop=True)  # Remove the first two rows (Ticker and Date rows)
            df['Ticker'] = ticker_value  # Add Ticker as a column

            df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker']

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index(['Ticker', 'Date'], inplace=True)
            df.sort_index(inplace=True)

            all_data.append(df)

    # Concatenate all dataframes without resetting the index
        self.df = pd.concat(all_data)
        print(self.df['Close'].dtype)

        self.df['Close'] = pd.to_numeric(self.df['Close'], errors='coerce')
        self.df['Volume'] = pd.to_numeric(self.df['Volume'], errors='coerce')

        # Calculate returns and rolling metrics
        self.df["Daily Return"] = self.df.groupby('Ticker')['Close'].pct_change()
        self.df["30D Rolling Avg"] = self.df.groupby('Ticker')['Close'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)
        self.df["14D EMA"] = self.df.groupby('Ticker')['Close'].ewm(span=14, adjust=False).mean().reset_index(level=0, drop=True)
        self.df["Volatility"] = self.df.groupby('Ticker')['Daily Return'].rolling(window=10, min_periods=1).std().reset_index(level=0, drop=True)

        self.df.groupby('Ticker')['Close'].ffill()
        self.df.groupby('Ticker')['Volume'].bfill()

        output_path = self.data_dir / "merged_stock_data.csv"
        self.df.to_csv(output_path)
        print(f"Transformed data saved to {output_path}")
        return self.df

class Analyzer:
    """Performs time-series analysis."""
    
    @staticmethod
    def detect_crossovers(df):
        # Initialize the 'Crossover' column with zeros
        df["Crossover"] = 0
        
        # Group by Ticker to avoid crossovers being calculated across tickers
        df["Crossover"] = df.groupby('Ticker').apply(
            lambda group: ((group["30D Rolling Avg"] > group["14D EMA"]) & 
                        (group["30D Rolling Avg"].shift(1) <= group["14D EMA"].shift(1))).astype(int) - 
                        ((group["30D Rolling Avg"] < group["14D EMA"]) & 
                        (group["30D Rolling Avg"].shift(1) >= group["14D EMA"].shift(1))).astype(int)
        ).reset_index(level=0, drop=True)
        
        # Forward-fill the crossover values to avoid issues at the first row
        df["Crossover"] = df.groupby('Ticker')['Crossover'].shift(1, fill_value=0)

        # Print out counts of each crossover state for validation
        unique_value_counts = df['Crossover'].value_counts()
        print(unique_value_counts)

        return df

class Visualizer:
    def __init__(self, color_palette):
        self.color_palette = color_palette
        
    """Handles data visualization."""

    @staticmethod
    def style_plot(func):
        """Wrapper to apply consistent styling to all plots."""
        def wrapper(*args, **kwargs):
            plt.style.use("seaborn-v0_8")
            fig, ax = plt.subplots(figsize=(14, 7))
            func(*args, ax=ax, **kwargs)

            ax.set_xlabel("Date", fontsize=14)
            ax.set_ylabel("Value", fontsize=14)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            fig.autofmt_xdate()

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

            plt.tight_layout()
            plt.show()
        return wrapper

    # @staticmethod
    @style_plot
    def plot_closing_prices(self, df, ax=None):
        sns.lineplot(data=df, x=df.index.get_level_values("Date"), y="Close", hue=df.index.get_level_values("Ticker"), ax=ax, palette=self.color_palette)
        ax.set_title("Stock Closing Prices", fontsize=16, weight='bold')

    # @staticmethod
    def plot_daily_returns(self, df):
        tickers = df.index.get_level_values("Ticker").unique()
        ncols = 3
        nrows = 2  # Fixed to 2 rows for a 2x3 grid

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), sharex=True, sharey=True)
        fig.suptitle("Daily Returns for Each Stock", fontsize=16, weight='bold')

        for i, ticker in enumerate(tickers):
            ax = axes[i // ncols, i % ncols]
            stock_df = df.xs(ticker, level="Ticker")
            daily_returns = stock_df["Daily Return"]

            sns.lineplot(data=daily_returns, ax=ax, color=self.color_palette[ticker])

            ax.set_title(f"{ticker} Daily Returns", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Daily Return (%)", fontsize=12)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

            # Ensure x-axis labels are shown for all plots
            for label in ax.get_xticklabels():
                label.set_visible(True)
                label.set_rotation(45)

            # Format y-axis as percentage without decimals (e.g., 20%)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        # Remove empty subplots if any
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes.flatten()[j])

        # Adjust layout to ensure labels are visible
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Increased bottom space slightly

        plt.show()

    # @staticmethod
    @style_plot
    def plot_moving_averages(self, df, ax=None):
        tickers = df.index.get_level_values("Ticker").unique()

        for ticker in tickers:
            stock_df = df.xs(ticker, level="Ticker")
            color = self.color_palette[ticker]

            ax.plot(stock_df.index, stock_df["30D Rolling Avg"], label=f"{ticker} 30-Day SMA", linestyle="--", linewidth=2, color=color)
            ax.plot(stock_df.index, stock_df["14D EMA"], label=f"{ticker} 14D EMA", linestyle=":", linewidth=2, color=color)

            positive_crossover = stock_df[stock_df["Crossover"] == 1]
            negative_crossover = stock_df[stock_df["Crossover"] == -1]

            ax.scatter(positive_crossover.index, positive_crossover["Close"], color=color, marker='^', label=f'{ticker} Positive Crossover', s=20, edgecolor='black', zorder=5)
            ax.scatter(negative_crossover.index, negative_crossover["Close"], color=color, marker='v', label=f'{ticker} Negative Crossover', s=20, edgecolor='black', zorder=5)

        ax.set_title("30 Day Rolling Avg & 14 Day EMA with Crossovers for Each Stock", fontsize=16, weight='bold')


class TestETL(unittest.TestCase):
    def test_data_loader(self):
        loader = DataLoader(["AAPL"], "2023-01-01", "2023-12-31")
        self.assertIsInstance(loader.tickers, list)
    
    def test_transformer(self):
        transformer = Transformer()
        sample_data = pd.DataFrame({"Close": [100, 102, 104], "Volume": [1000, None, 1200]}, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
        sample_data["Volume"].bfill(inplace=True)
        self.assertFalse(sample_data["Volume"].isna().any())
    
    def test_crossovers(self):
        sample_data = pd.DataFrame({
            "30D Rolling Avg": [100, 102, 98],
            "14D EMA": [101, 101, 99]
        })
        df = Analyzer.detect_crossovers(sample_data)
        self.assertListEqual(df["Crossover"].tolist(), [-1, 1, -1])
    
    def test_rolling_calculations(self):
        sample_data = pd.DataFrame({"Adj Close": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]},
                                   index=pd.date_range("2023-01-01", periods=10))
        sample_data["30D Rolling Avg"] = sample_data["Adj Close"].rolling(3).mean()
        self.assertAlmostEqual(sample_data["30D Rolling Avg"].iloc[-1], (112+114+116)/3)
    
    def test_daily_return(self):
        sample_data = pd.DataFrame({"Adj Close": [100, 105, 110]})
        sample_data["Daily Return"] = sample_data["Adj Close"].pct_change()
        self.assertAlmostEqual(sample_data["Daily Return"].iloc[1], 0.05)
        self.assertAlmostEqual(sample_data["Daily Return"].iloc[2], 0.047619)


def main(extract: bool = False, transform: bool = False, analyze: bool = False, visualize: bool = False, full: bool = False):
    tickers = ["AMD", "NVDA", "INTC", "MSFT", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    data_dir = Path(r"../data")

    if extract or full:
        loader = DataLoader(tickers, start_date, end_date, data_dir)
        loader.fetch_and_save()
    
    if transform or full:
        transformer = Transformer(data_dir)
        df = transformer.load_and_transform(tickers)
    
    if analyze or full:
        analyzer = Analyzer()
        df = pd.read_csv(
            data_dir / "merged_stock_data.csv",
            index_col=["Date", "Ticker"],
            parse_dates=["Date"], 
            date_format="%Y-%m-%d"
        )
        df = analyzer.detect_crossovers(df)
        df.to_csv(data_dir / "merged_stock_data_with_analysis.csv")
        print("Analysis completed.")
    
    if visualize:
        df = pd.read_csv(
            data_dir / "merged_stock_data_with_analysis.csv",
            index_col=["Date", "Ticker"], 
            parse_dates=["Date"], 
            date_format="%Y-%m-%d"
        )   
        visualizer = Visualizer(color_palette=dict(zip(tickers, plt.get_cmap('tab10', len(tickers)).colors)))
        visualizer.plot_closing_prices(df)
        visualizer.plot_daily_returns(df)
        visualizer.plot_moving_averages(df)

typer.run(main)


