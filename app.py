#streamlit app setup
import streamlit as st
import pandas as pd
import pickle

#transferring the models
with open('linear_model_all_types.pkl', 'rb') as file:
    linear_model = pickle.load(file)
with open('lasso_model_all_types.pkl', 'rb') as file:
    lasso_model = pickle.load(file)
with open('gradient_boosting_model.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

st.title("Calories Burned App")
#dropdown choices
modelChoice = st.selectbox("Select Prediction Model", [
    "Linear Regression",
    "Lasso Regression",
    "Gradient Boosting"
])
#input fields
age = st.number_input("Enter your Age", min_value=10, max_value=100, value=25)
weight = st.number_input("Enter your Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
session_duration = st.number_input("Session Duration (hours)", min_value=0.1, max_value=5.0, value=1.0)
workout_frequency = st.number_input("Workout Frequency (days/week)", min_value=1, max_value=7, value=3)
workout_type = st.selectbox("Select Workout Type", ['Cardio', 'HIIT', 'Strength', 'Yoga'])
#filling the metrics
if st.button("Predict Calories Burned"):
    inputData = {
        'Age': age,
        'Weight (kg)': weight,
        'Session_Duration (hours)': session_duration,
        'Workout_Frequency (days/week)': workout_frequency,
        'Workout_Type_Cardio': 0,
        'Workout_Type_HIIT': 0,
        'Workout_Type_Strength': 0,
        'Workout_Type_Yoga': 0
    }
    inputData[f'Workout_Type_{workout_type}'] = 1
    inputDf = pd.DataFrame([inputData])
    #to choose the dropdown models
    if modelChoice == "Linear Regression":
        prediction = linear_model.predict(inputDf)[0]
    elif modelChoice == "Lasso Regression":
        prediction = lasso_model.predict(inputDf)[0]
    elif modelChoice == "Gradient Boosting":
        prediction = gbm_model.predict(inputDf)[0]
    
    #output
    st.success(f"Estimated Calories Burned: {round(prediction/3, 2)} calories")
