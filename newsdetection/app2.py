import streamlit as st
import joblib
import re
import string

# --- LOAD THE SAVED FILES FROM YOUR ORIGINAL PROJECT ---
# Load the saved TF-IDF Vectorizer
vectorizer = joblib.load("vectorizer.jb")

# Load the saved Logistic Regression model
model = joblib.load("lr_model.jb")


# --- TEXT CLEANING FUNCTION (from your original notebook) ---
# This is needed to process the user's input in the same way the model was trained
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text.strip()


# --- USER INTERFACE ---
st.title("FAKE NEWS DETECTOR")
st.write("Enter a news article for checking :")

# Text input area for the user
news_input = st.text_area("News Article:", "")

# "Check News" button
if st.button("Check News"):
    # Check if the user has entered any text
    if news_input.strip():
        # 1. Clean the user's input text
        cleaned_input = clean_text(news_input)

        # 2. Transform the cleaned text using the loaded TF-IDF vectorizer
        transform_input = vectorizer.transform([cleaned_input])

        # 3. Use the loaded model to make a prediction
        prediction = model.predict(transform_input)

        # 4. Display the result
        if prediction[0] == 1:
            st.success("The News is Real !")
        else:
            st.error("the news is fake !")
    else:
        # If the text box is empty, show a warning
        st.warning("Please enter some text to analyze")
    