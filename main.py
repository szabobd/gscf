from utils.utils import DataLoader, Transformer, Analyzer, Visualizer, load_csv
import matplotlib.pyplot as plt
from pathlib import Path
import typer
from config import *

def main(extract: bool = False, transform: bool = False, analyze: bool = False, visualize: bool = False, full: bool = False):
    path = Path(data_dir)

    if extract or full:
        DataLoader(tickers, start_date, end_date, path).fetch_and_save()

    if transform or full:
        Transformer(path).load_and_transform(tickers)

    if analyze or full:
        Analyzer.analyze_data(path)

    if visualize:
        Visualizer.visualize(path, tickers)

if __name__ == "__main__":
    typer.run(main)