import streamlit as st
import pickle

# Load saved model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📧 Email Spam Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Transform input using vectorizer
        X_input = vectorizer.transform([user_input])

        # Predict using trained model
        prediction = model.predict(X_input)[0]

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

