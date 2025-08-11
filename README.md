# 📰 News Category Prediction Web App

A minimal yet professional **Streamlit** web application that predicts the category of a news article based on its text.

## 📌 Overview
This project uses a **Machine Learning model** trained on the BBC News dataset to classify news articles into different categories such as *business, politics, sport, tech,* and *entertainment*.

The app provides:
- A clean and professional UI
- Real-time predictions
- Preprocessing pipeline (tokenization, stopword removal, stemming)
- Simple deployment using Streamlit

> ⚠ **Note:** The dataset is old (around early 2000s, during George W. Bush presidency). This means predictions on modern news articles might not be highly accurate due to changes in language and topics.

---

## 🚀 How It Works
1. **Preprocessing** – The input text is:
   - Tokenized
   - Lowercased
   - Stopwords removed
   - Stemmed using PorterStemmer
2. **Vectorization** – Transformed into numerical features using a saved `CountVectorizer`.
3. **Prediction** – A trained model predicts the category.
4. **Display** – The category is shown with a simple animation in the UI.

---

## 📂 Project Structure
