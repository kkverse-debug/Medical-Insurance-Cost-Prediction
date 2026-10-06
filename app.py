import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("best_insurance_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Insurance Cost Prediction")
st.write("Enter the details below to predict the medical insurance cost.")

# User inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Prediction button
if st.button("Predict Insurance Cost"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Medical Insurance Cost: ${prediction:,.2f}"
    )