from utils.utils import DataLoader, Transformer, Analyzer, Visualizer
import matplotlib.pyplot as plt
from pathlib import Path
import typer
from config import *

def main(extract: bool = False, transform: bool = False, analyze: bool = False, visualize: bool = False, full: bool = False):
    path = Path(data_dir)

    if extract or full:
        DataLoader(tickers, start_date, end_date, path).fetch_api_and_save_to_csv()

    if transform or full:
        Transformer(path).load_and_transform(tickers)

    if analyze or full:
        Analyzer.analyze_data(path)

    if visualize:
        Visualizer.visualize(path, tickers)

if __name__ == "__main__":
    typer.run(main)