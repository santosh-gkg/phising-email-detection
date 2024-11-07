# Phishing Email Detection Project Explanation

The aim is to develop a NLP model to detect phishing emails. The code covers several steps, including data preprocessing, feature engineering, and model training. Let's break down the main components:

## 1. Data Loading and Initial Exploration

The project starts by importing necessary libraries and loading a dataset:

```python
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import Word, TextBlob

df = pd.read_csv("phisingDataset/CEAS_08.csv")
```

The dataset contain information about emails, including a "label" column (1 for phishing, 0 for legitimate).

## 2. Feature Engineering

### 2.1 Sender Analysis
- The code extracts the domain from the sender's email address.
- It creates binary features for specific email domains (edu, python, apache, loewis, gmail, org).

### 2.2 Sentiment Analysis
- Using NLTK's SentimentIntensityAnalyzer, the code calculates sentiment scores for both the subject and body of emails.
- A compound sentiment score is created by combining subject and body scores.

### 2.3 Spam Word Count
- The code loads a list of spam words from an Excel file.
- It then counts how many of these spam words appear in the combined subject and body text of each email.

### 2.4 Text Length
- The code calculates the length of the combined subject and body text.

### 2.5 URL Count
- The code includes a feature for the number of URLs in the email (presumably already present in the dataset).

## 3. Text Preprocessing

The subject and body text are preprocessed:
- Converted to lowercase
- Punctuation removed
- Combined into a single text field

## 4. Feature Selection

The code creates several feature sets:
1. `df_new`: A subset of engineered features
2. `X`: Combination of engineered features and TF-IDF vectors
3. `X_2`: Only TF-IDF vectors
4. `X_3`: Only engineered features (non-text)

## 5. Model Training and Evaluation

The project experiments with two types of models:

### 5.1 Logistic Regression
- Trained and evaluated on all three feature sets (X, X_2, X_3)
- The best performance is achieved with X_2 (only TF-IDF vectors)

### 5.2 Random Forest
- After determining that X_2 performs best, a Random Forest model is trained on this feature set

## 6. Visualization

The code includes a correlation heatmap to visualize relationships between engineered features.

## Key Observations

1. The project uses both text-based (TF-IDF) and engineered features.
2. Sentiment analysis and spam word counting are used as potential indicators of phishing.
3. The sender's domain is considered a potentially important feature.
4. TF-IDF vectors alone (X_2) seem to perform better than combining them with engineered features or using engineered features alone.
5. Both Logistic Regression and Random Forest models are tested, with Random Forest potentially performing better (though direct comparison isn't shown in the code).

## Potential Improvements

1. Feature importance analysis for the Random Forest model
2. Hyperparameter tuning for both models
3. Trying other algorithms (e.g., Gradient Boosting, SVM)
4. Cross-validation for more robust performance estimation
5. Addressing class imbalance if present in the dataset

This project demonstrates a comprehensive approach to email classification, combining traditional NLP techniques with custom feature engineering tailored to the phishing detection problem.
