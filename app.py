import streamlit as st
import pickle

# Load the trained pipeline (vectorizer + classifier together)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.title("📧 Email/SMS Spam Classifier")

st.write("This app predicts whether a message is **Spam** or **Not Spam**.")

# User input
user_input = st.text_area("✍️ Enter the message below:")

# Predict button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == 1:   # assuming spam = 1, ham = 0
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
