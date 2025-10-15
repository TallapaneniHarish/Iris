import streamlit as st
import numpy as np
import pickle


# Load trained model
with open("iris_model.sav", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Iris Species Prediction App 🌸")
st.write("Enter the measurements of the iris flower:")

# User input
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"The predicted class is: {prediction[0]}")
