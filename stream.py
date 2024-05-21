import joblib
import streamlit as st
import numpy as np

# Load the trained model
file_name = "finalized_model.sav"
loaded_model = joblib.load(file_name)

# Streamlit app title
st.title("Suicide Analysis")

# Markdown instruction
st.markdown("Write your thoughts. How are you doing? What are you thinking? How are your days going?")

# Removing the Streamlit banner at the bottom
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Text input for user
user_text = st.text_input("Text", key="user_text")

# Access the value and handle prediction
if user_text:
    try:
        # Preprocess the input text if necessary
        # For example, you might need to vectorize the text if your model expects it
        # processed_text = preprocess_text(user_text)  # Add this if you have a preprocessing step

        # Predict the result
        result = loaded_model.predict([user_text])  # Use processed_text if preprocessing is needed

        # Display the result
        st.write(f"Prediction: {result[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Run `streamlit run stream.py` to run the file
