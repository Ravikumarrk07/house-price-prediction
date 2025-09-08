import pandas as pd

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop Loan_ID as it is not useful for prediction
    df = df.drop('Loan_ID', axis=1)

    # Fill missing values (updated to avoid chained assignment warnings)
    df.loc[:, 'Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df.loc[:, 'Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df.loc[:, 'Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df.loc[:, 'Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df.loc[:, 'LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df.loc[:, 'Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df.loc[:, 'Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Convert categorical to numerical
    df.replace({
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
        'Loan_Status': {'Y': 1, 'N': 0}
    }, inplace=True)

    # Convert 'Dependents' to numeric (handle '3+')
    df.loc[:, 'Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    return df
