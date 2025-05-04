import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and feature columns
model = joblib.load("churn_stack_model.pkl")
with open("feature_columns.json") as f:
    feature_cols = json.load(f)

# Title
st.title("🔮 Customer Churn Prediction")
st.markdown("""
Enter basic customer info to predict churn risk using a stacked machine learning model. 
The model estimates the probability that a customer will stop using the service based on their attributes.
""")

# Sidebar - Simplified input form
st.sidebar.header("Customer Info")

def user_input():
    input_data = {}
    input_data['Tenure Months'] = st.sidebar.slider(
        'Tenure (Months)', 0, 72, 12,
        help="How long has the customer been with us?"
    )
    input_data['Monthly Charges'] = st.sidebar.slider(
        'Monthly Charges', 0, 200, 70,
        help="How much is the customer billed per month?"
    )
    input_data['Contract'] = st.sidebar.selectbox(
        'Contract Type', [0, 1, 2],
        format_func=lambda x: ['Month-to-Month', 'One Year', 'Two Year'][x],
        help="What is the customer's contract term?"
    )
    input_data['Internet Service'] = st.sidebar.selectbox(
        'Internet Service', [0, 1, 2],
        format_func=lambda x: ['No', 'DSL', 'Fiber'][x],
        help="Type of internet service the customer uses."
    )
    input_data['Tech Support'] = st.sidebar.selectbox(
        'Tech Support', [0, 1, 2],
        format_func=lambda x: ['No', 'Yes', 'No Internet'][x],
        help="Whether the customer has technical support access."
    )
    input_data['Dependents'] = st.sidebar.selectbox(
        'Has Dependents', [0, 1],
        help="Does the customer have dependents?"
    )
    return pd.DataFrame([input_data])

input_df = user_input()

# Add default values for missing features
default_input = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)
default_input.update(input_df)

# Compute engineered features
default_input['Is_New_Customer'] = (default_input['Tenure Months'] < 6).astype(int)
default_input['Has_Contract'] = (default_input['Contract'] != 0).astype(int)
default_input['High_Monthly_Cost'] = (default_input['Monthly Charges'] > 70).astype(int)
default_input['Risky_Services_Count'] = (
    (default_input['Streaming TV'] == 2).astype(int) +
    (default_input['Streaming Movies'] == 2).astype(int) +
    (default_input['Tech Support'] == 0).astype(int) +
    (default_input['Online Security'] == 0).astype(int)
)
default_input['Engaged_Customer_Score'] = (
    (default_input['Dependents'] == 1).astype(int) +
    (default_input['Tech Support'] == 2).astype(int) +
    (default_input['Online Backup'] == 2).astype(int) +
    (default_input['Tenure Months'] > 12).astype(int) +
    (default_input['Contract'] == 2).astype(int)
)

# Ensure all required columns are present
for col in feature_cols:
    if col not in default_input.columns:
        default_input[col] = 0

# Predict
if st.button("Predict Churn"):
    prob = model.predict_proba(default_input)[0][1]
    churn_label = "High Risk 🔴" if prob > 0.7 else ("Medium Risk 🟠" if prob > 0.4 else "Low Risk 🟢")

    st.subheader("Prediction Result")
    st.metric(label="Churn Probability", value=f"{prob:.2%}")
    st.success(f"Customer Churn Risk: {churn_label}")

    st.markdown("""
    This means the model estimates there's a {:.2f}% chance this customer will stop using the service.
    Based on this probability, they are classified as **{}**.
    """.format(prob * 100, churn_label))

    st.info("""
    **How to use this result:**
    - Use high risk scores to flag customers for retention campaigns.
    - Medium scores may require targeted communication.
    - Low scores suggest stable, loyal customers.
    """)