import pandas as pd
import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression

# Load the pre-trained Logistic Regression model from pickle file
with open('titanic_logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title('Prediction of Titanic Survival')

# Collect user input for a new prediction
Pclass = st.selectbox("Class", [1, 2, 3])
Age = st.number_input("Age")
SibSp = st.number_input("Siblings/Spouses aboard")
Parch = st.number_input("Parents/Children aboard")
Fare = st.number_input("Fare")
Sex = st.selectbox("Sex", ["male", "female"])
Embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert user input for encoding
Sex_male = 1 if Sex == 'male' else 0
Embarked_Q = 1 if Embarked == 'Q' else 0
Embarked_S = 1 if Embarked == 'S' else 0

# Create a DataFrame for the new data
new_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Sex_male': [Sex_male],
    'Embarked_Q': [Embarked_Q],
    'Embarked_S': [Embarked_S]
})

# Make prediction
if st.button('Predict Survival'):
    prediction = model.predict(new_data)
    survival_status = "Survived" if prediction[0] == 1 else "not survived"
    st.write(f'The passenger will survive: {survival_status}')
