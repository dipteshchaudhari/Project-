import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# --- PAGE CONFIGURATION ---
# This sets the title and icon that appear in the browser tab.
st.set_page_config(
    page_title="Fake News Detector 2.0",
    page_icon="📰",
    layout="wide"
)

# --- LOAD YOUR SAVED MODEL AND TOKENIZER ---
# This is the most important part. We load the "brain" of our app.
# @st.cache_resource tells Streamlit to only do this once, making the app fast.
@st.cache_resource
def load_assets():
    """Load all the necessary files for the application to work."""
    try:
        # 1. Load your final, fine-tuned Logistic Regression model
        final_model = joblib.load('final_model.joblib')

        # 2. Load the BERT tokenizer you saved
        bert_tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')

        # 3. Load the base BERT model (this is needed to process new text)
        bert_model = TFBertModel.from_pretrained('bert-base-uncased', use_safetensors=False)
        return final_model, bert_tokenizer, bert_model
    except Exception as e:
        st.error(f"Error loading necessary files: {e}")
        st.error("Please make sure 'final_model.joblib' and the 'bert_tokenizer' folder are in the same directory as this app.")
        return None, None, None

# Load the assets when the app starts
model, tokenizer, bert_model = load_assets()

# --- HELPER FUNCTION TO PROCESS NEW TEXT ---
# This function takes the user's text and turns it into a "meaning barcode".
def get_bert_embedding_for_prediction(text_input):
    """Takes raw text and converts it into a BERT embedding."""
    inputs = tokenizer(text_input, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = bert_model(inputs)
    embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embedding.numpy()

# --- STYLING (For a professional look) ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; color: #00aaff; text-align: center; margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2rem; color: #e0e0e0; text-align: center; margin-bottom: 40px;
    }
    .stButton>button {
        background-color: #00aaff; color: white; border-radius: 8px; padding: 12px 24px;
        font-size: 1.1rem; border: none; width: 100%; transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0088cc;
    }
    .prediction-box {
        padding: 25px; border-radius: 10px; text-align: center;
        font-size: 1.8rem; font-weight: bold; margin-top: 30px;
    }
    .real-news {
        background-color: #155724; color: #d4edda; border: 1px solid #c3e6cb;
    }
    .fake-news {
        background-color: #721c24; color: #f8d7da; border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# --- USER INTERFACE ---
st.markdown('<p class="main-header">Fake News Detector 2.0</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by BERT and a Fine-Tuned Logistic Regression Model</p>', unsafe_allow_html=True)

# Use columns to center the main content and make it look clean
col1, col2, col3 = st.columns([1, 2.5, 1])

with col2:
    # A text box for the user to paste news into.
    news_input = st.text_area("Enter News Article Text:", "", height=250, label_visibility="collapsed", placeholder="Paste the full text of the news article here...")

    # The "Analyze News" button.
    if st.button("Analyze News"):
        if model is not None and news_input.strip() != "":
            with st.spinner("Analyzing... This may take a moment."):
                # 1. Process the user's input to get the embedding.
                transformed_input = get_bert_embedding_for_prediction(news_input)

                # 2. Use your model to predict FAKE (0) or REAL (1).
                prediction = model.predict(transformed_input)
                
                # 3. **GET THE CONFIDENCE SCORE (PROBABILITY)**
                probability = model.predict_proba(transformed_input)

                # 4. **DISPLAY THE RESULT WITH THE CONFIDENCE SCORE**
                if prediction[0] == 1:
                    # If the prediction is REAL (1), get the confidence for the REAL class
                    confidence = probability[0][1] * 100
                    st.markdown(f'<div class="prediction-box real-news">Prediction: REAL News<br>(Confidence: {confidence:.2f}%)</div>', unsafe_allow_html=True)
                else:
                    # If the prediction is FAKE (0), get the confidence for the FAKE class
                    confidence = probability[0][0] * 100
                    st.markdown(f'<div class="prediction-box fake-news">Prediction: FAKE News<br>(Confidence: {confidence:.2f}%)</div>', unsafe_allow_html=True)
        elif news_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            st.error("Application models could not be loaded. Cannot make a prediction.")
