import streamlit as st
import os
import nltk
import pickle
import time

tokenizer = nltk.tokenize
stemmer = nltk.stem.porter.PorterStemmer()

current_dir = os.path.dirname(os.path.realpath(__file__))

# Load stopwords
with open(os.path.join(current_dir, 'Models', 'stopwords.txt')) as stopwords_file:
    stopwords = stopwords_file.readlines()
stopwords = [line.strip() for line in stopwords]
nltk_stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(nltk_stopwords)
stopwords = set(stopwords)  # for faster lookup

# Load vectorizer, label encoder, and model
with open(os.path.join(current_dir, 'Models', 'lion_v.h5'), 'rb') as f:
    v = pickle.load(f)

with open(os.path.join(current_dir, 'Models', 'lion_le.h5'), 'rb') as f:
    le = pickle.load(f)

with open(os.path.join(current_dir, 'Models', 'lion_model.h5'), 'rb') as f:
    model = pickle.load(f)

# -------------------------------
# Function to preprocess text
# -------------------------------
def preprocess_text(text):
    tokens = tokenizer.word_tokenize(text)
    filtered_tokens = [w.lower() for w in tokens if w.isalpha() and w.lower() not in stopwords]
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    return " ".join(stemmed_tokens)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="News Category Prediction",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stTextArea textarea {
        background-color: #f9f9f9;
        border: 1px solid #cccccc;
        border-radius: 10px;
        font-size: 16px;
        color: #333333;
    }
    .predict-btn button {
        background-color: #0066cc !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .result {
        font-size: 20px;
        font-weight: 600;
        color: #0066cc;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #0066cc;'>📰 News Category Prediction</h1>", unsafe_allow_html=True)
st.write("Enter a news article text below and click **Predict** to see its category.")

news_text = st.text_area(
    "News Text",
    placeholder="Enter the news text here",
    height=150
)

if st.button("Predict", key="predict_btn"):
    if news_text.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing the news..."):
            processed_text = preprocess_text(news_text)  # Apply same preprocessing
            news_vectorized = v.transform([processed_text])
            prediction = model.predict(news_vectorized)
            category = le.inverse_transform(prediction)[0]
            time.sleep(0.8)

        st.markdown(
            f"<div class='result'>✨ The category of this news is: <span style='color:#ff6600'>{category}</span></div>",
            unsafe_allow_html=True
        )
