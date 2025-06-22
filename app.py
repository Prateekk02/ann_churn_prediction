import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from tensorflow.keras.models import load_model

# Load the model
model = load_model("models/model.keras")

# Load label encoder, one-hot encoder, scaler
with open('models/label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('models/one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geography = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit application
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_hr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox("Is an Active Member?", [0, 1])

# Predict button
if st.button("Predict"):
    # Input data preparation
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_hr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Encoding geography
    geo_encoded = one_hot_encoder_geography.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder_geography.get_feature_names_out(['Geography'])
    )

    # Combine one-hot encoded data with input data
    encoded_input = pd.concat([input_data, geo_encoded_df], axis=1)

    # Scale the data
    scaled_input = scaler.transform(encoded_input)

    # Predict churn
    prediction_churn = model.predict(scaled_input)
    prediction_prob = prediction_churn[0][0]

    # Display result
    if prediction_prob > 0.5:
        st.write("The customer is **likely to churn**.")
    else:
        st.write("The customer is **not likely to churn**.")
