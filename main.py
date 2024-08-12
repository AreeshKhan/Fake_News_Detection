import pickle
import streamlit as st
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Load models and vectorizer
vectoriser_model = pickle.load(open('models/vectorization.pkl', 'rb'))
GradientBoostingClassifier_model = pickle.load(open('models/GradientBoostingClassifier.pkl', 'rb'))
DecisionTree_model = pickle.load(open('models/DecisionTreeClassifier.pkl', 'rb'))

# Title of the Streamlit app
st.title("Fake News Detection App")

# Text input for user
st.header("From Dataset only:")
st.header("Enter the news(From Dataset only) text to classify:")
user_input = st.text_area("News Text", "")

# Function to preprocess the input text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)  # Replace non-word characters with spaces
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
    return text

# Button to make predictions
if st.button("Predict"):
    # Preprocess the text
    processed_text = wordopt(user_input)
    # Vectorize the text
    new_data_vectorised = vectoriser_model.transform([processed_text])
    # Make predictions
    result1 = GradientBoostingClassifier_model.predict(new_data_vectorised)
    result2 = DecisionTree_model.predict(new_data_vectorised)
    final_pred = bool(result1) and bool(result2)

    # Display the result
    if final_pred:
        st.success("The news is classified as **True**.")
    else:
        st.error("The news is classified as **Fake**.")
