import sys
import streamlit as st

# Menampilkan versi Python yang sedang berjalan
st.write("Python version:", sys.version)

# Rest of your app code
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.h5')

# Streamlit app
st.title("Phishing URL Detection")
st.sidebar.header("Navigation")
pages = ["Home", "EDA", "Predict"]
choice = st.sidebar.radio("Go to", pages)

if choice == "Home":
    st.write("""
    # Welcome to the Phishing URL Detection App
    This app uses a machine learning model to predict whether a URL is phishing or safe.
    """)

elif choice == "EDA":
    st.write("# Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Overview")
        st.write(data.head())
        
        st.write("Statistics")
        st.write(data.describe())
        
        st.write("Missing Values")
        st.write(data.isnull().sum())

elif choice == "Predict":
    st.write("# Predict Phishing URLs")
    
    # Input single URL for prediction
    url_input = st.text_input("Enter URL for prediction", "")
    if url_input:
        # Feature extraction from the input URL
        url_features = {
            'url_length': len(url_input),
            'num_dots': url_input.count('.'),
            'num_slashes': url_input.count('/')
        }
        features = pd.DataFrame([url_features])
        
        # Make prediction for the single URL
        prediction = model.predict(features)
        prediction_label = "Phishing" if prediction[0] == 1 else "Safe"
        
        st.write(f"The URL is: **{url_input}**")
        st.write(f"Prediction: **{prediction_label}**")
    
    # File upload for batch prediction
    uploaded_file = st.file_uploader("Upload your dataset for prediction (CSV)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(data.head())

        # Feature extraction for batch prediction
        data['url_length'] = data['URL'].apply(len)
        data['num_dots'] = data['URL'].apply(lambda x: x.count('.'))
        data['num_slashes'] = data['URL'].apply(lambda x: x.count('/'))
        
        # Select features for prediction
        features = data[['url_length', 'num_dots', 'num_slashes']]
        
        # Make predictions
        predictions = model.predict(features)
        data['Prediction'] = predictions
        st.write("Predictions:")
        st.write(data[['URL', 'Prediction']])

        # Save predictions to CSV
        result_file = "predictions.csv"
        data.to_csv(result_file, index=False)
        st.download_button(
            label="Download Predictions",
            data=open(result_file, "rb"),
            file_name=result_file,
            mime="text/csv",
        )
