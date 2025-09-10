import streamlit as st
import pickle

# Load trained pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📧 Email/SMS Spam Classifier")

user_input = st.text_area("Enter the message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
