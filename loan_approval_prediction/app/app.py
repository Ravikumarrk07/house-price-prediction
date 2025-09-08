import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('models/loan_model.pkl')

st.title("🏦 Loan Approval Prediction App")
st.markdown("Fill in the applicant details below:")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [360.0, 120.0, 180.0, 240.0, 300.0, 84.0, 60.0, 36.0, 12.0])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs to model format
input_data = {
    'Gender': 1 if gender == "Male" else 0,
    'Married': 1 if married == "Yes" else 0,
    'Dependents': 3 if dependents == "3+" else int(dependents),
    'Education': 1 if education == "Graduate" else 0,
    'Self_Employed': 1 if self_employed == "Yes" else 0,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': 2 if property_area == "Urban" else 1 if property_area == "Semiurban" else 0
}

input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Loan Status"):
    prediction = model.predict(input_df)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader("Prediction Result:")
    st.success(result)
