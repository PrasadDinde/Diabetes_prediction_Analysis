pip install scikit-learn
pip install xgboost
pip install lightgbm
# Add more libraries as needed

import sklearn
import streamlit as st
import pandas as pd
import pickle
import os

# Load the model
if os.path.exists('classification_model.pkl'):
    with open('classification_model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    st.error("Model file not found! Make sure 'classification_model.pkl' is in the correct location.")

# Function to make predictions
def make_prediction(outcome):
    prediction = model.predict(outcome)
    probability = model.predict_proba(outcome)[:, 1]  # Probability of being diabetic
    return prediction, probability

# Streamlit UI
st.title("Diabetes Analysis Prediction")
st.subheader("Enter Human Body Information")

Pregnancies = st.number_input("Enter Pregnancies", min_value=0, max_value=20, step=1)
Glucose = st.number_input("Enter Glucose", min_value=30, max_value=300, step=1)
BloodPressure = st.number_input("Enter Blood Pressure", min_value=20, max_value=150, step=1)
SkinThickness = st.number_input("Enter Skin Thickness", min_value=5, max_value=100, step=1)
Insulin = st.number_input("Enter Insulin", min_value=14.00, max_value=900.00, step=0.01)
BMI = st.number_input("Enter BMI", min_value=15.00, max_value=70.00, step=0.1)
DiabetesPedigreeFunction = st.number_input("Enter Diabetes Pedigree Function", min_value=0.01, max_value=3.00, value=0.5, step=0.01)
Age = st.number_input("Enter Age", min_value=15, max_value=100, step=1)

input_data = pd.DataFrame({
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
    'Age': [Age]
})

st.write("Input Data:")
st.write(input_data)

# Prediction button
if st.button('Predict'):
    try:
        prediction, probability = make_prediction(input_data)
        st.write(f"Diabetes Prediction: {prediction[0]}")
        st.write(f"Prediction Probability: {probability[0] * 100:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.info("Note: 0 means not Diabetic, 1 means Diabetic")
