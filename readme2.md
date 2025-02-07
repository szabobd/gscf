


# Yahoo Finance Data Processing Tool

## Overview
This Python-based data processing tool fetches stock data from Yahoo Finance API, processes, analyzes and visualizes it. The project was written in the *3.11.0* version of Python, and uses packages pandas, matplotlib, seaborn, typer and yfinance in the process.

## Components
- **main.py**: Orchestrates the process and adds a command line interface to interact with the script.
- **tests**: Directory contains the following test files, containing tests related to the classes used in the pipeline.
    * `data_loader_test.py`
    * `transformer_test.py`
    * `analyzer_test.py`
    * `visualizer_test.py`
- **my_utils**: Directory contains the classes used in the process, separated in to the following files:
    * `data_loader.py`
    * `transformer.py`
    * `analyzer.py`
    * `visualizer.py`
- **config.py**: Contains parameters to configure the pipeline:
    * `tickers`
    * `start_date`
    * `end_date`
    * `data_dir`
    * `filename_merged`
    * `filename_with_analysis`
- **data_dir**: Folder defined in `config.json` (default is `data`) the contains `.csv` files created by the pipeline during the process


## Configuration
The `config.json` file serves as the configuration file for the data processing tool. Keep in mind that the parameters are designed according to the task description. Altering them does not necesssarily break the code, but can result in malfunctions and errors, as the code and tests were written according to the scope defined in the task description. The parameters are:
- `tickers`: Five unique labels to identify the stocks to be processed. 
- `start_date`: Selects the start date for the period being analyzed.
- `end_date`: Selects the end date for the period being analyzed.
- `data_dir`: The path of the directory where the `.csv` files created during the process are stored.
- `filename_merged`: The name of the `.csv` file that contains the dataframe merged and formatted from the separate stock data `.csv` files extracted from `yfinance`.
- `filename_with_analysis`: The name of the `.csv` that contains the dataframe with the values from the time series analysis in *step 5*.

## Usage
1. **Installation**: Ensure Python is installed on your system. Extract the contents of the `homework_SZB.zip` file to the desired location in your folder structure, enter the folder where the contents of `homework_SZB.zip` were extracted to from your CLI and install the required dependencies using `pip install -r requirements.txt`.
2. **Configuration**: As parameters are set to the pre-described values, you can leave `config.py` file as it is.
3. **Execution**: Enter the folder containing `main.py`from you CLI. To run the script, use the `python main.py --{parameter}` command. The first four commands should be ran in the given order for the pipeline to function correctly when ran for the first time. There is also the option to add more than one flags to the command like `python main.py --extract --transform`.
The five arguments:
- `extract`: Runs the extraction part of the script, downloads the ticker data from Yahoo Finance and saves them as separate `.csv` files
- `transform`: Applies the necessary transformations and adds columns containing 30 day Moving Average and 14 day Exponential Moving Average, Daily Return and Volatility values
- `analyze`: Finds crossovers between 30 day Moving Average and 14 day Exponential Moving Average
- `visualize`: Visualizes closing price, daily return and crossovers for the selected tickers
- `full`: Runs the whole pipeline 
There is also the option to add more than one flags to the command like `python main.py --extract --transform`

5. **Output**:The output JSON files are saved in the `result_json_sets/` directory. Each new process output is stored in a separate directory named according to the user-configured `experiment_id`. Within these directories, you can find two JSON files: one for the training set and one for the test set.
6. **Testing**: Run `pytest` command from the root directory to execute the test files in the `tests` folder.

## Known limitations
1. The `config.py`should not be modified in general, as the script is written and tested to operate with the parameters given. 
2. However, it should function with `start_date` and `end_date` between the given dates 2020-01-01 and 2023-12-31, apart from the cases of selecting a time window smaller than the lengths given for the rolling averages (30 and 14 day).
3. Checking other tickers should also work, as long as the number of tickers is equal to 5.


---