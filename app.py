import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ✅ Load pre-trained model and scaler
model = load_model("model_final.h5")
sc = joblib.load("scaler.pkl")

# ✅ Streamlit UI setup
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🏦 Customer Churn Prediction App")

st.markdown("""
Enter customer details and get a **predicted churn probability**.
""")

with st.form("input_form"):
    geography = st.selectbox("🌍 Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    credit_score = st.slider("💳 Credit Score", 350, 850, 600)
    age = st.slider("🎂 Age", 18, 100, 40)
    tenure = st.slider("📅 Tenure (years)", 0, 10, 3)
    balance = st.number_input("💰 Balance", min_value=0.0, max_value=1_000_000.0, value=60000.0, format="%.2f")
    num_products = st.selectbox("🛍️ Number of Products", [1, 2, 3, 4])
    has_credit_card = st.selectbox("💳 Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("🔥 Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("💵 Estimated Salary", min_value=0.0, max_value=1_000_000.0, value=50000.0, format="%.2f")
    submitted = st.form_submit_button("🚀 Predict")

if submitted:
    # ✅ Encode categorical variables
    geo = [0, 0, 0]
    if geography == "France":
        geo = [1, 0, 0]
    elif geography == "Germany":
        geo = [0, 1, 0]
    elif geography == "Spain":
        geo = [0, 0, 1]

    gen = 1 if gender == "Male" else 0
    has_cc = 1 if has_credit_card == "Yes" else 0
    active = 1 if is_active_member == "Yes" else 0

    # ✅ Combine features and scale
    features = geo + [credit_score, gen, age, tenure, balance, num_products, has_cc, active, estimated_salary]
    features = np.array(features).reshape(1, -1)
    scaled_features = sc.transform(features)

    # ✅ Predict churn probability
    prob = float(model.predict(scaled_features)[0][0])
    prediction = "Churn" if prob > 0.5 else "Stay"

    # ✅ Display result
    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob:.3f}")
    if prob > 0.5:
        st.error("🚨 Customer likely to CHURN")
    else:
        st.success("✅ Customer likely to STAY")
