import streamlit as st
import pandas as pd
import pickle
import sklearn
import os

with open('Machine Learning/Supervised learning/Project_Predictive_Analysis_Diabetes/classification_model.pkl', 'rb') as f:
    model = pickle.load(f)


def make_prediction(outcome):
    prediction = model.predict(outcome)
    return prediction

st.title("Diabetes Analysis Prediction")
st.subheader("Enter Human Body information")

Pregnancies	= st.number_input("Enter Pregnancies ",min_value=1,max_value=20,step=1)
Glucose	= st.number_input("Enter Glucose ",min_value=30,max_value=300,step=1)
BloodPressure = st.number_input("Enter BloodPressure ",min_value=20,max_value=150,step=1)
SkinThickness = st.number_input("Enter SkinThickness ",min_value=5,max_value=100,step=1)
Insulin	= st.number_input("Enter Insulin ",min_value=14.00,max_value=900.00,step=0.01)
BMI	= st.number_input("Enter BMI ",min_value=15.00,max_value=70.00,step=0.1)
DiabetesPedigreeFunction = st.number_input("Enter DiabetesPedigreeFunction ", min_value=0.01, max_value=3.00, value=0.5, step=0.01)
Age	= st.number_input("Enter Age ",min_value=15,max_value=100,step=1)

input_data = pd.DataFrame({'Pregnancies':[Pregnancies],
                           'Glucose':[Glucose],
                           'BloodPressure':[BloodPressure],
                           'SkinThickness':[SkinThickness],
                           'Insulin':[Insulin],
                           'BMI':[BMI],
                           'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],
                           'Age':[Age]})

st.write("Input Data:")
st.write(input_data)

if st.button('Predict'):
    try:
        prediction = make_prediction(input_data)
        st.write(f"Diabetes Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.info("Note :0 means not Diabetes | 1 means Diabetes")
