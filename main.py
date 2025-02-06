from my_utils.data_loader import DataLoader
from my_utils.transformer import Transformer
from my_utils.analyzer import Analyzer
from my_utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from pathlib import Path
import typer
from config import *

def main(extract: bool = False, transform: bool = False, analyze: bool = False, visualize: bool = False, full: bool = False):
    path = Path(data_dir)

    if extract or full:
        DataLoader(tickers, start_date, end_date, path).fetch_api_and_save_to_csv()

    if transform or full:
        Transformer(path).load_and_transform(tickers, filename_merged)

    if analyze or full:
        Analyzer.analyze_data(path, filename_merged, filename_with_analysis)

    if visualize:
        visualizer = Visualizer(color_palette=dict(zip(tickers, plt.get_cmap('tab10', len(tickers)).colors)))
        visualizer.visualize(path, filename_with_analysis)

if __name__ == "__main__":
    typer.run(main)