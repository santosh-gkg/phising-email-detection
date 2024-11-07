# Streamlit app for phishing email detection
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# Load pre-trained models and vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("phishing_logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)
with open("phishing_random_model.pkl", "rb") as f:
    random_model = pickle.load(f)


# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define helper functions
def get_compound_score(text):
    return sia.polarity_scores(text)['compound']

def count_spam_words(text, spam_words):
    text = text.lower()
    count = 0
    for word in spam_words["SPAM WORDS"]:
        if word in text:
            count += 1
    return count

# Load spam words for feature engineering
spam_words = pd.read_excel("phisingDataset/Spam_Words.xlsx")
spam_words["SPAM WORDS"] = spam_words["SPAM WORDS"].str.lower().str.replace('[^\w\s]', '')

# Streamlit app layout
st.title("Phishing Email Detection App")

# Input fields for email data
sender = st.text_input("Sender Email")
subject = st.text_area("Email Subject")
body = st.text_area("Email Body")

if st.button("Predict"):
    # Feature engineering
    subject = subject.lower()
    body = body.lower()
    subject_body = subject + " " + body
    subject_body = ''.join(e for e in subject_body if e.isalnum() or e.isspace())
    
    compound_subject = get_compound_score(subject)
    compound_body = get_compound_score(body)
    compound_score = compound_subject + compound_body
    spam_count = count_spam_words(subject_body, spam_words)
    text_length = len(subject_body)

    # Email domain-based features
    edu_mail = 1 if 'edu' in sender else 0
    python_mail = 1 if 'python' in sender else 0
    apache_mail = 1 if 'apache' in sender else 0
    loewis_mail = 1 if 'loewis' in sender else 0
    gmail_mail = 1 if 'gmail' in sender else 0
    org_mail = 1 if 'org' in sender else 0

    # TF-IDF transformation
    tfidf_features = tfidf_vectorizer.transform([subject_body])

    # Create feature vector for prediction
    sparse_features = csr_matrix([[compound_subject, spam_count, text_length, edu_mail,
                                   python_mail, apache_mail, loewis_mail, gmail_mail, org_mail]])
    X_input = tfidf_features

    # Predict using the loaded model
    logistic_prediction = logistic_model.predict(X_input)
    random_prediction = random_model.predict(X_input)

    # prediction of logistic regression model
    if logistic_prediction[0] == 1:
        st.write("The email is likely **Phishing** according to the Logistic Regression model.")
    else:
        st.write("The email is likely **Legitimate** according to the Logistic Regression model.")
    
    # prediction of random forest model
    if random_prediction[0] == 1:
        st.write("The email is likely **Phishing** according to the Random Forest model.")
    else:
        st.write("The email is likely **Legitimate** according to the Random Forest model.")

