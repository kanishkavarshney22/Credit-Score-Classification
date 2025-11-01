import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "artifacts/model.pkl"

st.set_page_config(page_title="Credit Score Classification", page_icon="💳")

st.title("💳 Credit Score Classification App")
st.write("Predict customer credit score based on financial information.")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Please train the model first using credit_score_train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

st.header("Enter Customer Details")
age = st.number_input("Age", 18, 100, 25)
income = st.number_input("Annual Income", 10000, 200000, 50000)
loan_amount = st.number_input("Loan Amount", 1000, 50000, 10000)
credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1, 0])
dependents = st.slider("Number of Dependents", 0, 5, 0)

if st.button("Predict Credit Score"):
    debt_to_income = loan_amount / income
    features = pd.DataFrame([{
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_history": credit_history,
        "dependents": dependents,
        "debt_to_income": debt_to_income
    }])
    pred = model.predict(features)[0]
    st.success(f"Predicted Credit Score: **{pred}**")