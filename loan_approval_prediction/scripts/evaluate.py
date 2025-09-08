import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib
from preprocess import preprocess_data

# Load and preprocess data
df = preprocess_data('data/train.csv')
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split test data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = joblib.load('models/loan_model.pkl')

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
