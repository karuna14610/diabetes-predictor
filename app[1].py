
import streamlit as st
import numpy as np
import pickle
import os  # for file check

# ✅ Show all files in the current directory (debugging)
st.write("Files in app folder:", os.listdir())

# Load model and scaler
model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# App Title
st.title("🩺 Diabetes Prediction System")
st.write("Enter the health details below:")

# Input fields
preg = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose Level', min_value=0)
bp = st.number_input('Blood Pressure', min_value=0)
skin = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin', min_value=0)
bmi = st.number_input('BMI')
dpf = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age', min_value=0)

# Collect input
input_data = [preg, glucose, bp, skin, insulin, bmi, dpf, age]

# Predict button
if st.button('Predict'):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("⚠️ The person is Diabetic")
    else:
        st.success("✅ The person is Not Diabetic")
