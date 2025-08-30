import streamlit as st
import numpy as np
import pickle

# Load the trained model & scaler (if you saved both together)
model, scaler = pickle.load(open("trained_model.sav", "rb"))

# Streamlit app
st.title("💳 Credit Card Default Prediction App")
st.write("This app predicts whether a customer will default on their credit card payment next month.")

# Collect user inputs for the 8 important features
limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0, value=50000, step=1000)
pay_0 = st.number_input("Payment Status (PAY_0)", min_value=-2, max_value=8, value=0)
pay_2 = st.number_input("Payment Status (PAY_2)", min_value=-2, max_value=8, value=0)
bill_amt1 = st.number_input("Bill Amount (BILL_AMT1)", min_value=0, value=10000, step=1000)
pay_amt1 = st.number_input("Previous Payment Amount (PAY_AMT1)", min_value=0, value=5000, step=1000)
pay_amt2 = st.number_input("Payment Amount (PAY_AMT2)", min_value=0, value=5000, step=1000)
pay_amt3 = st.number_input("Payment Amount (PAY_AMT3)", min_value=0, value=5000, step=1000)
marriage = st.selectbox(
    "Marital Status (MARRIAGE)",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x]
)

# Prepare input for model (order must match training dataset columns)
# Make sure the order here is SAME as the 'selected_columns' you trained with
features = np.array([[pay_0, pay_amt2, limit_bal, pay_2, pay_amt3, bill_amt1, pay_amt1, marriage]])



# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ The customer is likely to DEFAULT next month.")
    else:
        st.success("✅ The customer is NOT likely to default next month.")
