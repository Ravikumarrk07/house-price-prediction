import pandas as pd
import joblib
from preprocess import preprocess_data

# Load model
model = joblib.load('models/loan_model.pkl')

# Example input (can be modified)
input_data = {
    'Gender': 1,
    'Married': 1,
    'Dependents': 0,
    'Education': 1,
    'Self_Employed': 0,
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 0,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0,
    'Property_Area': 1
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
prediction = model.predict(input_df)[0]
result = 'Approved' if prediction == 1 else 'Rejected'

print("Predicted Loan Status:", result)
