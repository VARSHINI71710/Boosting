import streamlit as st
import numpy as np
import pickle

# Load the trained model & scaler (if you saved both together)
model, scaler = pickle.load(open("trained_model.sav", "rb"))


st.title("💳 Credit Card Default Prediction")

st.markdown("Enter customer details to predict if they will default next month:")


PAY_0 = st.number_input("Payment Status PAY_0", min_value=-2, max_value=8, value=0)
PAY_AMT2 = st.number_input("Previous Payment PAY_AMT2", min_value=0, value=5000)
LIMIT_BAL = st.number_input("Credit Limit (LIMIT_BAL)", min_value=1000, value=50000)
PAY_2 = st.number_input("Payment Status PAY_2", min_value=-2, max_value=8, value=0)
PAY_AMT3 = st.number_input("Payment Amount PAY_AMT3", min_value=0, value=5000)
BILL_AMT1 = st.number_input("Bill Amount BILL_AMT1", min_value=0, value=10000)
PAY_AMT1 = st.number_input("Previous Payment PAY_AMT1", min_value=0, value=5000)
MARRIAGE = st.selectbox("Marital Status", ["Married", "Single", "Others"])
MARRIAGE_val = 1 if MARRIAGE == "Married" else 2 if MARRIAGE == "Single" else 3


if st.button("Predict"):
    
    input_array = np.array([PAY_0, PAY_AMT2, LIMIT_BAL, PAY_2, PAY_AMT3, BILL_AMT1, PAY_AMT1, MARRIAGE_val]).reshape(1, -1)
    
    
    input_scaled = scaler.transform(input_array)
    
    
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    
    
    st.subheader("Prediction Result:")
    st.write("🔹 Default" if prediction == 1 else "🔹 No Default")
    st.write(f"🔹 Probability of Default: {proba:.2f}")
