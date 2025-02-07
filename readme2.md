# Yahoo Finance Data Processing Tool

## 📌 Overview
This Python-based data processing tool fetches stock data from Yahoo Finance API, processes, analyzes, and visualizes it. The project is built with **Python 3.11.0** and utilizes the following key packages:

- `pandas`
- `matplotlib`
- `seaborn`
- `typer`
- `yfinance`

## 📁 Project Structure
```
├── main.py              # CLI for running the pipeline
├── tests/               # Contains unit tests
│   ├── data_loader_test.py
│   ├── transformer_test.py
│   ├── analyzer_test.py
│   ├── visualizer_test.py
├── my_utils/            # Core utility scripts
│   ├── data_loader.py
│   ├── transformer.py
│   ├── analyzer.py
│   ├── visualizer.py
├── config.py            # Configuration settings
├── data_dir/            # Directory for generated CSV files
└── README.md            # Project documentation
```

## ⚙️ Configuration
The pipeline is configured via `config.py`. The following parameters are used:

| Parameter              | Description                                      |
|------------------------|--------------------------------------------------|
| `tickers`             | List of five stock symbols to process           |
| `start_date`          | Start date for the analysis period              |
| `end_date`            | End date for the analysis period                |
| `data_dir`            | Directory where CSV files are stored            |
| `filename_merged`     | Merged CSV filename                              |
| `filename_with_analysis` | CSV file containing processed analysis data  |

> **Note:** Altering these parameters may cause unintended errors as the code is tailored to a predefined scope.

## 🚀 Usage
### 1️⃣ Installation
Ensure Python is installed, then run:
```bash
pip install -r requirements.txt
```

### 2️⃣ Running the Pipeline
Navigate to the project folder and execute `main.py` with one or more commands:
```bash
python main.py --extract --transform
```
Available commands:

| Command       | Description |
|--------------|-------------|
| `--extract`  | Downloads stock data from Yahoo Finance and saves as CSV |
| `--transform` | Applies transformations, including Moving Averages |
| `--analyze`   | Identifies moving average crossovers |
| `--visualize` | Generates stock data visualizations |
| `--full`      | Runs the entire pipeline |

### 3️⃣ Running Tests
Execute all unit tests using:
```bash
pytest
```

## 📊 Output
All processed data is saved in the `data_dir/` directory. The pipeline generates:  

- **Five separate `.csv` files**—one for each selected ticker, containing raw extracted stock data.  
- **Two additional `.csv` files**:  
  - `filename_merged.csv` — Stores the merged and formatted data from all tickers.  
  - `filename_with_analysis.csv` — Contains the analyzed data, including calculated indicators and crossovers.  


## ⚠️ Known Limitations
1. **Config constraints:** The script is optimized for five stock tickers and a date range from `2020-01-01` to `2023-12-31`.  
2. **Short time frames:** A time window smaller than 30 or 14 days may affect rolling average calculations.  
3. **Ticker count:** The script may not function correctly if the number of tickers differs from five.  

---

💡 **Tip:** For best results, keep the configuration as-is unless you're familiar with modifying the code accordingly.
