import streamlit as st
import pickle
import numpy as np

model=pickle.load(open("model.pkl"),"rb")

st.title("My SMS Classifier app with streamlit")

feature1 = st.number_input("Enter feature 1", min_value=0, max_value=100, value=50)
feature2 = st.number_input("Enter feature 2", min_value=0, max_value=100, value=30)

if st.button("Predict"):
    prediction = model.predict(np.array([[feature1, feature2]]))
    st.success(f"Prediction: {prediction[0]}")