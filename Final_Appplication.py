import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb

# Load the trained model
model = pickle.load(open('lgb_model.pkl', 'rb'))

def predict_customer_churn(features):
    prediction = model.predict([features])[0]
    return "The customer is likely to stay." if prediction == 0 else "The customer is likely to leave."

def main():
    st.title('Customer Churn Prediction')

    # Collect user inputs
    gender = st.selectbox("Customer's Gender", ('Male', 'Female'))
    senior_citizen = st.checkbox('Is the customer a senior citizen?')
    partner = st.checkbox('Does the customer have a partner?')
    dependents = st.checkbox('Does the customer have any dependents?')
    tenure = st.number_input("Customer's tenure in months", min_value=0)
    phone_service = st.checkbox('Does the customer have phone service?')
    multiple_lines = st.checkbox('Does the customer have multiple lines?')
    internet_service = st.selectbox("Type of internet service", ('DSL', 'Fiber optic', 'No'))
    online_security = st.selectbox("Online security", ('Yes', 'No', 'No internet service'))
    online_backup = st.selectbox("Online backup", ('Yes', 'No', 'No internet service'))
    device_protection = st.selectbox("Device protection", ('Yes', 'No', 'No internet service'))
    tech_support = st.selectbox("Tech support", ('Yes', 'No', 'No internet service'))
    streaming_tv = st.selectbox("Streaming TV", ('Yes', 'No', 'No internet service'))
    streaming_movies = st.selectbox("Streaming movies", ('Yes', 'No', 'No internet service'))
    contract = st.selectbox("Type of contract", ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.checkbox('Does the customer have paperless billing?')
    payment_method = st.selectbox("Payment method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.number_input("Customer's monthly charges", min_value=0.0)
    total_charges = st.number_input("Customer's total charges", min_value=0.0)

    # Map inputs to model expected format
    feature_vector = [
        1 if gender == 'Male' else 0,
        1 if senior_citizen else 0,
        1 if partner else 0,
        1 if dependents else 0,
        tenure,
        1 if phone_service else 0,
        1 if multiple_lines else 0,
        {'DSL': 0, 'Fiber optic': 1, 'No': 2}[internet_service],
        {'Yes': 1, 'No': 0, 'No internet service': 2}[online_security],
        {'Yes': 1, 'No': 0, 'No internet service': 2}[online_backup],
        {'Yes': 1, 'No': 0, 'No internet service': 2}[device_protection],
        {'Yes': 1, 'No': 0, 'No internet service': 2}[tech_support],
        {'Yes': 1, 'No': 0, 'No internet service': 2}[streaming_tv],
        {'Yes': 1, 'No': 0, 'No internet service': 2}[streaming_movies],
        {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[contract],
        1 if paperless_billing else 0,
        {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}[payment_method],
        monthly_charges,
        total_charges
    ]

    # Extend the feature_vector to match the expected number of features
    additional_features = [0] * (26 - len(feature_vector))  # You can adjust these default values based on your model's training
    feature_vector.extend(additional_features)

    if st.button('Predict Churn'):
        result = predict_customer_churn(feature_vector)
        st.success(result)

if __name__ == "__main__":
    main()
