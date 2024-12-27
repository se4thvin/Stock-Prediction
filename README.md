# S&P500 Stock Market Index Price Prediction

**Project Overview**

In this project, we'll predict the price of the S&P500 stock market index.

**Project Steps**

* Download data using the yfinance package
* Create an initial machine learning model and estimate accuracy
* Build a backtesting engine to more accurately measure accuracy
* Improve the accuracy of the model

## Features
- **Automatic Data Fetching**: Retrieves S&P 500 historical data from 1980 to the current date using the `yfinance` library.
- **Feature Engineering**: Dynamically computes technical indicators like:
  - `Close_Ratio_2`, `Close_Ratio_5`, `Close_Ratio_60`, `Close_Ratio_250`, and `Close_Ratio_1000`.
- **Model Training and Evaluation**:
  - Implements **Time-Series Cross-Validation** for robust evaluation.
  - Tracks key metrics including Precision, Recall, F1-Score, and ROC-AUC.
- **Interactive Visualizations**:
  - **Time-Series Chart**: Shows historical S&P 500 trends.
  - **Feature Importance**: Highlights the most impactful features using an interactive bar chart.
- **Buy/Sell Recommendations**:
  - Provides actionable insights with confidence levels for the latest available data.

**File overview:**

* `market_prediction.ipynb` - a Jupyter notebook that contains all of the code.

## Live Application
Check out the live Streamlit app:  
[Stock Prediction App](https://stock-prediction-z3na7gm9kcjwcxkaazaqyq.streamlit.app/)



**Installation**

To follow this project, please install the following locally:

* JupyterLab
* Python 3.8+
* Python packages:
    * pandas
    * yfinance
    * scikit-learn

**Data**

We'll download all of the data during the project, using the yfinance package.
