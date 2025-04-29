import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

model = joblib.load('loan_approval_model_dt.pkl')

columns = ['Type of Account', 'Account History', 'Reason for the Loan',
       'Loan Amount', 'Account Savings', 'Employment History',
       'Individual Stauts', 'Other Loans', 'Security / Collateral', 'Age',
       'Residence Status', 'Job', 'Completed Other loan?']

with open('model_features_dt.json') as f:
    columns_to_keep = json.load(f)
    
st.title("Loan Approval Prediction App")

st.markdown("Add details below: ")

# create input

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Marial Status", ["Yes", "No", "Divorced"])
Age = st.number_input("Age", min_value = 18, max_value = 100, step=1)
Employment_history = st.selectbox("Employment History", ['0-2 Years', '2-5 Years', '5-7 Years', '7+ Years', 'Unemployed'])
Job = st.selectbox("Job", ['Not Employed', 'Professional / Management', 'Services','Skilled Labor'])
Residence_status = st.selectbox("Residence Status", ['Free', 'House Owner', 'Renting'])
Account_Saving = st.selectbox("Account Savings", ['0-200', '1000+', '200-500', '500-1000', 'No Data'])
Loan_amount = st.number_input("Loan Amount", min_value= 0)
Other_loans = st.selectbox("Other Loans", ['No', 'Yes'])
Security_Collateral = st.selectbox("Security / Collateral", ['No Security', 'Property - Real Estate', 'Savings Account','Vehicle'])
Account_type = st.selectbox("Type of Account", ['No Data', 'Type A', 'Type B', 'Type C'])
Reason_for_loan = st.selectbox("Reason for the Loan", ['Advance Edu/training', 'All other', 'Buying a New Car','Buying a Used Car', 'Home Devices', 'Home furniture','Learning / Edu purposes', 'Renovation', 'Support for Business','TV'])
Account_history = st.selectbox("Account History", ['Average (known delays)', 'Good', 'critical'])
Completed_Other_loan = Other_loans

if Married=="No":
    Indivudual_status = Gender
else:
    Indivudual_status = Married



# Completed Other loan? = if other loan==true   yes or no

input_dict = {
    'Individual Stauts': Indivudual_status,
    'Type of Account': Account_type,
    'Account History': Account_history,
    'Reason for the Loan': Reason_for_loan,
    'Loan Amount': Loan_amount,
    'Account Savings': Account_Saving,
    'Employment History': Employment_history,
    'Other Loans': Other_loans,
    'Security / Collateral': Security_Collateral,
    'Age ': Age,
    'Residence Status': Residence_status,
    'Job ': Job,
    'Completed Other loan?': Completed_Other_loan,
 
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)

for col in columns_to_keep:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[columns_to_keep]


if st.button("Check Loan approval"):
    result = model.predict(input_encoded)
    if result == 1:
        st.success("Congratulations! Your loan is likely to be approved.")
    else:
        st.error("Sorry, your loan is unlikely to be approved.")