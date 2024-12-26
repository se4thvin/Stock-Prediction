import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import plotly.express as px

# App Title
st.title("S&P 500 Stock Prediction")

# Sidebar Inputs
st.sidebar.title("Settings")
start_date = st.sidebar.date_input("Start Date", value=datetime(1980, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
n_estimators = st.sidebar.slider("Number of Estimators", 100, 1000, 500, step=100)
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.6, 0.05)

# Function to fetch S&P 500 data
@st.cache
def fetch_sp500_data(start_date, end_date):
    try:
        sp500_data = yf.download('^GSPC', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        sp500_data.reset_index(inplace=True)
        return sp500_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch data
st.write("Fetching the latest S&P 500 data...")
data = fetch_sp500_data(start_date, end_date)
if data is None:
    st.stop()

# Display the first few rows of the dataset
st.write("Dataset Preview:")
st.dataframe(data.head())

# Feature Engineering
st.write("Processing data...")
if 'Close' in data.columns:
    data['Close_Ratio_2'] = data['Close'].pct_change(2)
    data['Close_Ratio_5'] = data['Close'].pct_change(5)
    data['Close_Ratio_60'] = data['Close'].rolling(60).mean() / data['Close']
    data['Close_Ratio_250'] = data['Close'].rolling(250).mean() / data['Close']
    data['Close_Ratio_1000'] = data['Close'].rolling(1000).mean() / data['Close']
    data.dropna(inplace=True)
else:
    st.error("The dataset does not contain the 'Close' column, which is required to calculate important features.")
    st.stop()

# Define features
important_features = ['Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', 'Close_Ratio_250', 'Close_Ratio_1000']

# Add Target column for demonstration
data['Target'] = (data['Close'] > data['Close'].shift(1)).astype(int)

# Model Training and Validation
st.write("Training and validating the model...")
tscv = TimeSeriesSplit(n_splits=5)
precisions, recalls, f1_scores, roc_aucs = [], [], [], []

for train_index, test_index in tscv.split(data):
    train, test = data.iloc[train_index], data.iloc[test_index]
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
    model.fit(train[important_features], train["Target"])
    preds = model.predict_proba(test[important_features])[:, 1]
    preds = (preds >= threshold).astype(int)

    precisions.append(precision_score(test["Target"], preds))
    recalls.append(recall_score(test["Target"], preds))
    f1_scores.append(f1_score(test["Target"], preds))
    roc_aucs.append(roc_auc_score(test["Target"], preds))

st.write(f"Average Precision: {np.mean(precisions):.2f}")
st.write(f"Average Recall: {np.mean(recalls):.2f}")
st.write(f"Average F1-Score: {np.mean(f1_scores):.2f}")
st.write(f"Average ROC-AUC: {np.mean(roc_aucs):.2f}")

# Feature Importance Visualization
st.write("Feature Importances:")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": important_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importances")
st.plotly_chart(fig)

# Footer
st.write("Developed by Sethvin Nanayakkara")