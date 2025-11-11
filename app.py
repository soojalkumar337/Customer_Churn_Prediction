
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Customer Churn Prediction', layout='centered')
st.title("📱 Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error('Model files not found. Please run the training notebook/script to generate churn_model.pkl and scaler.pkl.')
    st.stop()

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0,1])
partner = st.selectbox("Partner", ["Yes","No"])
dependents = st.selectbox("Dependents", ["Yes","No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=240, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=500.0)

gender_val = 1 if gender == 'Male' else 0
partner_val = 1 if partner == 'Yes' else 0
dependents_val = 1 if dependents == 'Yes' else 0

input_df = pd.DataFrame([
    [gender_val, senior_citizen, partner_val, dependents_val, tenure, monthly_charges, total_charges]
], columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MonthlyCharges', 'TotalCharges'])

# Align with training features if available
if hasattr(scaler, 'feature_names_in_'):
    cols = scaler.feature_names_in_.tolist()
    input_df = input_df.reindex(columns=cols, fill_value=0)

X_scaled = scaler.transform(input_df)
prob = model.predict_proba(X_scaled)[:,1][0]
pred = model.predict(X_scaled)[0]

if st.button('Predict'):
    if pred == 1:
        st.error(f'⚠️ Predicted: Churn (Probability: {prob*100:.2f}%)')
    else:
        st.success(f'✅ Predicted: Stay (Probability: {prob*100:.2f}%)')
