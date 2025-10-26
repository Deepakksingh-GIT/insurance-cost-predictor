# InsuranceCost_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")
st.title("💊 Medical Insurance Cost Prediction")
st.markdown("Enter your details below to estimate your annual insurance cost.")

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("insurance_cost_model.pkl")
    model_loaded = True
except Exception as e:
    st.error("Could not load model. Ensure 'insurance_cost_model.pkl' exists in the folder.")
    st.stop()

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Enter your details")
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0, 0.1)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# -----------------------------
# Predict Button
# -----------------------------
if st.sidebar.button("Predict"):
    # Create input dataframe
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])

    # One-hot encode categorical variables (same as training)
    input_encoded = pd.get_dummies(input_df)
    
    # Ensure all 11 columns exist (fill missing columns with 0)
    expected_columns = [
        'age', 'bmi', 'children',
        'sex_female', 'sex_male',
        'smoker_no', 'smoker_yes',
        'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'
    ]
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[expected_columns]  # reorder columns

    # Predict
    pred = model.predict(input_encoded)[0]
    st.subheader("💰 Estimated Annual Insurance Cost")
    st.success(f"${pred:,.2f}")
    st.info("This is an estimate; actual premium may vary based on the insurer.")

st.markdown("---")
st.markdown("Developed by Deepak Kumar Singh — Medical Insurance Cost Prediction")
