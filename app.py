import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# Load trained model
# ============================================================

folder = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(folder, "insurance_model.pkl")
)


# ============================================================
# Page settings
# ============================================================

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰"
)

st.title("💰 Insurance Cost Predictor")

st.write(
    "Enter the details to estimate insurance expenses."
)


# ============================================================
# Inputs
# ============================================================

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)


bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)


sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)


smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)


children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)


region = st.selectbox(
    "Region",
    ["Southeast", "Other"]
)


# ============================================================
# Convert inputs to model features
# ============================================================

isfemale = 1 if sex == "Female" else 0

issmoker = 1 if smoker == "Yes" else 0

issoutheast = 1 if region == "Southeast" else 0


# ============================================================
# Prediction
# ============================================================

if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "isfemale": isfemale,
        "issmoker": issmoker,
        "children": children,
        "region_southeast": issoutheast
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Insurance Cost: ₹{prediction:,.2f}"
    )