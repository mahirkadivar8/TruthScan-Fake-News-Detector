import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup

# Load model and vectorizer
with open('model/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def fetch_article_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article = ' '.join([p.text for p in paragraphs])
        return article
    except:
        return None

def predict_news(text):
    input_tfidf = tfidf.transform([text])
    prediction = model.predict(input_tfidf)[0]
    confidence = model.predict_proba(input_tfidf)[0]
    return prediction, confidence

# App title
st.title("TruthScan - AI Fake News Detector")
st.write("Enter a news article or URL to check if it is Real or Fake!")

# Tabs
tab1, tab2 = st.tabs(["Paste Article", "Enter URL"])

# Tab 1 - Article
with tab1:
    news_text = st.text_area("Paste news article here:", height=200)
    if st.button("Check Article"):
        if news_text.strip() == "":
            st.warning("Please enter some text!")
        else:
            prediction, confidence = predict_news(news_text)
            if prediction == 0:
                st.error(f"FAKE NEWS! Confidence: {confidence[0]*100:.2f}%")
            else:
                st.success(f"REAL NEWS! Confidence: {confidence[1]*100:.2f}%")

# Tab 2 - URL
with tab2:
    url_input = st.text_input("Paste news URL here:")
    st.info("Note: Some websites may block article fetching. Try different news URLs if one does not work.")
    if st.button("Check URL"):
        if url_input.strip() == "":
            st.warning("Please enter a URL!")
        else:
            with st.spinner("Fetching article..."):
                article = fetch_article_from_url(url_input)
            if article is None or len(article) < 100:
                st.error("Could not fetch article from this URL!")
            else:
                prediction, confidence = predict_news(article)
                if prediction == 0:
                    st.error(f"FAKE NEWS! Confidence: {confidence[0]*100:.2f}%")
                else:
                    st.success(f"REAL NEWS! Confidence: {confidence[1]*100:.2f}%")