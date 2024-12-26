import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# App Title
st.title("S&P 500 Stock Prediction")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Preprocessing
    important_features = ['Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', 'Close_Ratio_250', 'Close_Ratio_1000']
    if all(f in data.columns for f in important_features):
        scaler = MinMaxScaler()
        data[important_features] = scaler.fit_transform(data[important_features])

        # Model Training
        st.write("Training the model...")
        train = data.iloc[:-100]
        test = data.iloc[-100:]
        model = RandomForestClassifier(n_estimators=500, random_state=1)
        model.fit(train[important_features], train["Target"])
        preds = model.predict_proba(test[important_features])[:, 1]

        # Adjust Threshold
        threshold = st.slider("Select Prediction Threshold", 0.0, 1.0, 0.6, 0.05)
        preds = (preds >= threshold).astype(int)

        # Evaluate
        precision = precision_score(test["Target"], preds)
        st.write(f"Model Precision: {precision:.2f}")

        # Feature Importance
        st.write("Feature Importances:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(important_features)), importances[indices], align="center")
        plt.xticks(range(len(important_features)), [important_features[i] for i in indices], rotation=90)
        st.pyplot(plt)
    else:
        st.write("The dataset must include the following columns:", important_features)

# Footer
st.write("Developed by [Your Name]")