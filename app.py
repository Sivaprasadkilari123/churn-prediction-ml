import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

model = joblib.load("churn_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("📉 Customer Churn Prediction")
st.write("Telecom customer churn prediction using XGBoost (trained with SMOTE for class imbalance). Enter customer details to predict churn risk.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

with col2:
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 150.0, 65.0)
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)
    support_calls = st.slider("Support Calls (last period)", 0, 10, 1)

if st.button("Predict Churn Risk", type="primary"):
    input_dict = {
        "gender": gender, "senior_citizen": senior_citizen, "partner": partner,
        "dependents": dependents, "tenure": tenure, "contract": contract,
        "internet_service": internet_service, "online_security": online_security,
        "tech_support": tech_support, "paperless_billing": paperless_billing,
        "payment_method": payment_method, "monthly_charges": monthly_charges,
        "total_charges": total_charges, "support_calls": support_calls
    }
    input_df = pd.DataFrame([input_dict])
    for col, le in encoders.items():
        if col in input_df.columns and col != "churn":
            input_df[col] = le.transform(input_df[col])
    input_df = input_df[feature_columns]

    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ High Churn Risk — {proba*100:.1f}% probability")
    else:
        st.success(f"✅ Low Churn Risk — {proba*100:.1f}% probability")

    st.progress(float(proba))

    st.subheader("Top Churn Drivers (Overall Model)")
    fi = pd.read_csv("feature_importance.csv", index_col=0)
    st.bar_chart(fi.head(6))

st.divider()
st.caption("Model: XGBoost | Trained on synthetic telecom customer data (3,000 customers, 20% churn rate) with SMOTE class balancing")
