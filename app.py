import streamlit as st
import joblib
import numpy as np

model = joblib.load('models/insurance_model.pkl')

st.title("🏥 Insurance Cost Prediction")
st.write("Enter your information to predict your insurance cost.")

age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 53.0, 25.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

reg_northeast = 1 if region == "northeast" else 0
reg_northwest = 1 if region == "northwest" else 0
reg_southeast = 1 if region == "southeast" else 0
reg_southwest = 1 if region == "southwest" else 0

if st.button("Predict Cost"):
    input_data = np.array([[age, sex, bmi, children, smoker,
                            reg_northeast, reg_northwest, reg_southeast, reg_southwest]])
    result = model.predict(input_data)
    st.success(f"Estimated Insurance Cost: ${result[0]:,.2f}")
