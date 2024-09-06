import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
#from wordcloud import WordCloud, STOPWORDS
import re

# Preprocessing and evaluation
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from nltk.tokenize import word_tokenize

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
import streamlit as st

df =pd.read_csv("C://Users//mansi//Documents//UCC//IS6611//final_project//Undersampled Dataset.csv")
# Define your custom stopword list
custom_stopwords = {'nothing', 'patient', 'would', 'could'}

# Combine custom stopword list with NLTK's English stopwords
stop_words = set(stopwords.words('english')).union(custom_stopwords)

#stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def text_preprocessing(text):
    if isinstance(text, list):
        # Convert list of strings into a single string
        text = ' '.join(text)
     # Check if the text is a float, and convert it to a string if necessary
    if isinstance(text, float):
        text = str(text)
        
    # Define patterns
    url_pattern = r"(http://[^ ]*|https://[^ ]*|www\.[^ ]*)"
    user_pattern = r"@[^\s]+"
    entity_pattern = r"&[^ ]+;"
    neg_contraction = r"n't\b"
    non_alpha = r"[^a-z]"
    punctuation_pattern = r"[^A-Za-z0-9]"
    whitespace_pattern = r"[ +]"

    # Lowercase the text
    cleaned_text = text.lower()

    # Replace negation contractions
    cleaned_text = re.sub(neg_contraction, " not", cleaned_text)

    # Remove URLs, user mentions, HTML entities, punctuation, whitespace
    cleaned_text = re.sub(url_pattern, " ", cleaned_text)
    cleaned_text = re.sub(user_pattern, " ", cleaned_text)
    cleaned_text = re.sub(entity_pattern, " ", cleaned_text)
    cleaned_text = re.sub(punctuation_pattern, " ", cleaned_text)
    cleaned_text = re.sub(whitespace_pattern, " ", cleaned_text)

    # Remove non-alphabetic characters
    cleaned_text = re.sub(non_alpha, " ", cleaned_text)

    # Tokenize the cleaned text
    tokens = word_tokenize(cleaned_text)

    # Define stopwords outside the function
    stop_words = set(stopwords.words('english'))

    # Lemmatize the words
    lemmatized_tokens = [lemmatizer.lemmatize(word, 'v') for word in tokens if word not in stop_words]

    return lemmatized_tokens
df['Preprocessed_Feedback'] = df['Feedback'].apply(text_preprocessing)  
X_train, X_test, y_train, y_test = train_test_split(df['Preprocessed_Feedback'], df['Sentiment'], test_size=0.2)
# Convert each list of strings into a single string
X_train = [' '.join(doc) for doc in X_train]
X_test = [' '.join(doc) for doc in X_test]

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer()

# Fit and transform training data
train_tfidf_matrix = tfidf.fit_transform(X_train)

# Transform testing data
test_tfidf_matrix = tfidf.transform(X_test)

pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
models = [BernoulliNB(),
          RandomForestClassifier(),
          SVC(),
          LogisticRegression()]
accuracy = []
precision = []
recall = []
f1_score = []

for model in models:
    model.fit(train_tfidf_matrix, y_train)
    y_pred = model.predict(test_tfidf_matrix)

    # Accuracy
    accuracy.append(model.score(test_tfidf_matrix, y_test))

    # Precision, Recall, F1 Score
    report = classification_report(y_test, y_pred, output_dict=True)
    precision.append(report['macro avg']['precision'])
    recall.append(report['macro avg']['recall'])
    f1_score.append(report['macro avg']['f1-score'])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix for", type(model).__name__)
    print(cm)
    print()

# DataFrame with accuracy, precision, recall, and F1 score
models_name = ['BernoulliNB', 'RandomForestClassifier', 'SVC','LogisticRegression']
results = pd.DataFrame({
    'Model': models_name,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score
})
results
log = SVC()
log.fit(train_tfidf_matrix, y_train)

pred = log.predict(test_tfidf_matrix)
pickle.dump(log, open('ml_model.pkl', 'wb'))
# Load the TF-IDF vectorizer and the trained SVC model
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
log = pickle.load(open('ml_model.pkl', 'rb'))

# Preprocess the feedback data
processed_feedbacks = df['Feedback'].apply(text_preprocessing)

# Convert the preprocessed feedbacks into strings
processed_feedbacks_str = processed_feedbacks.apply(lambda x: ' '.join(x))

# Transform the preprocessed feedback data using the TF-IDF vectorizer
transformed_feedback = tfidf.transform(processed_feedbacks_str)

# Predict the sentiment for each feedback
predicted_sentiment = log.predict(transformed_feedback)

# Add the predicted sentiment to the DataFrame
df['Predicted Sentiment'] = predicted_sentiment

# Display the DataFrame with feedback and predicted sentiment
print(df[['Feedback', 'Predicted Sentiment']])

# Save the DataFrame with the new column to a CSV file
df.to_csv('output.csv', index=False)

# Calculate overall scores
overall_positive_score = (df['Predicted Sentiment'] == 'positive').mean()
overall_negative_score = (df['Predicted Sentiment'] == 'negative').mean()
overall_neutral_score = (df['Predicted Sentiment'] == 'neutral').mean()

print("Overall Positive Score:", overall_positive_score)
print("Overall Negative Score:", overall_negative_score)
print("Overall Neutral Score:", overall_neutral_score)