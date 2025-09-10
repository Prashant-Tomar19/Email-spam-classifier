import streamlit as st
import pickle
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data (only first time, will be cached by Streamlit)
nltk.download("stopwords")
nltk.download("punkt")

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer if you saved it separately
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except:
    vectorizer = None  # If not saved, maybe your model pipeline includes it

# App title
st.title("📧 Email/SMS Spam Classifier")

# User input
user_input = st.text_area("Enter the message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform input text
        if vectorizer:
            transformed_text = vectorizer.transform([user_input])
        else:
            transformed_text = [user_input]  # In case model itself handles preprocessing

        # Predict
        prediction = model.predict(transformed_text)[0]

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
